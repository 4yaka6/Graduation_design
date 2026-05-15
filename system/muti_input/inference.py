import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from deep_translator import GoogleTranslator
from torchvision import models, transforms
from transformers import RobertaModel, RobertaTokenizer

INPUT_IMAGE_DIR = "uploadimage"  # 图片文件夹
INPUT_TEXT_FILE = "uploadtext.txt"  # 文本文件路径

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(project_root, "models")

IMAGE_WEIGHTS = os.path.join(MODEL_DIR, "image.pth")
TEXT_WEIGHTS = os.path.join(MODEL_DIR, "text.pth")

# 心理学 VA 映射表 (用于图像概率加权和分类转换)
EMOTIONS = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]
VA_MAP = torch.tensor([
    [-0.67, 0.75],  # Anger
    [-0.63, 0.45],  # Frustrated
    [0.0, 0.0],  # Neutral
    [0.7, 0.55],  # Happiness
    [0.85, 0.85],  # Excited
    [-0.75, -0.45]  # Sadness
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 模型结构类 =================

class ImageModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        # 内部加权计算 VA: (B, 6) @ (6, 2) -> (B, 2)
        va_coords = torch.mm(probs, VA_MAP.to(logits.device))
        return logits, va_coords


class TextModel(nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dim=256, num_classes=6):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.classifier_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        self.regressor_head = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 取最后四层平均
        cat_hidden = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
        cls_output = cat_hidden[:, 0, :]
        logits = self.classifier_head(cls_output)
        va = self.regressor_head(cls_output)
        return logits, va


# ================= 2. 处理工具类 =================

class ProcessImage:
    def __init__(self, input_dir="uploadimage"):
        self.input_dir = input_dir
        # 加载人脸检测分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _build_tensor_from_bgr(self, img_bgr):
        if img_bgr is None:
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            padding = int(w * 0.1)
            y1 = max(0, y - padding)
            y2 = min(img_bgr.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(img_bgr.shape[1], x + w + padding)
            face_roi = img_bgr[y1:y2, x1:x2]
            face_resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
            final_img = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            print("成功提取完整人脸并缩放至 224x224")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            final_img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            print(" 未检测到人脸，已将原图整体缩放")

        pil_img = Image.fromarray(final_img)
        return self.transform(pil_img).unsqueeze(0)

    def process_path(self, img_path):
        if not img_path or not os.path.exists(img_path):
            return None

        try:
            img_bgr = cv2.imread(img_path)
            return self._build_tensor_from_bgr(img_bgr)
        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            return None

    def get_processed_tensor(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"在 {self.input_dir} 路径下未找到图片文件")
            return None

        img_path = os.path.join(self.input_dir, files[0])

        try:
            img_bgr = cv2.imread(img_path)
            tensor = self._build_tensor_from_bgr(img_bgr)

            if os.path.exists(img_path):
                os.remove(img_path)
            return tensor

        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            return None

class ProcessText:
    def __init__(self, model_name="roberta-base"):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=False,
        )
        self.translator = GoogleTranslator(source='auto', target='en')

    def _translate_to_english(self, raw_text):
        try:
            eng_text = self.translator.translate(raw_text)
            print(f"   [翻译结果]: {eng_text}")
            return eng_text
        except Exception as e:
            print(f"文本翻译失败，回退原文输入: {e}")
            return raw_text

    def get_processed_input(self, raw_text):
        if not raw_text:
            return None

        eng_text = self._translate_to_english(raw_text)
        encoded = self.tokenizer.encode_plus(
            eng_text, add_special_tokens=True, max_length=128,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']


# ================= 3. 融合与推理引擎 =================

class FusionEngine:
    def __init__(self, img_model, txt_model, img_weights, txt_weights):
        self.img_model = img_model.to(DEVICE)
        self.txt_model = txt_model.to(DEVICE)
        self.img_model.load_state_dict(torch.load(img_weights, map_location=DEVICE))
        self.txt_model.load_state_dict(torch.load(txt_weights, map_location=DEVICE))
        self.img_model.eval()
        self.txt_model.eval()

        # 保持你的优势阵营定义
        self.txt_dominant = ["Frustrated", "Sadness"]  # 文本更准确
        self.img_dominant = ["Anger", "Happiness", "Excited"]      # 图像更直观

    def _apply_probability_threshold(self, probs):
        """
        逻辑 A：概率重分布
        如果最大概率 > 0.6，则将其余情绪权重降为 1/4 并重新归一化
        """
        new_probs = probs.copy()
        max_idx = np.argmax(new_probs)
        if new_probs[max_idx] > 0.6:
            mask = np.ones(len(new_probs), dtype=bool)
            mask[max_idx] = False
            new_probs[mask] *= 0.25  # 其他权重降至 1/4
            new_probs /= np.sum(new_probs)  # 重新归一化
        return new_probs

    def _calculate_weighted_va(self, probs):
        """
        逻辑 B：基于VA_MAP 进行期望计算
        """
        va_map_np = VA_MAP.cpu().numpy() if torch.is_tensor(VA_MAP) else VA_MAP
        return np.dot(probs, va_map_np)

    def run(self, img_tensor, txt_input):
        res = {"img": None, "txt": None}

        with torch.no_grad():
            # 1. 图像支路推理
            if img_tensor is not None:
                logits, _ = self.img_model(img_tensor.to(DEVICE))
                raw_probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                processed_probs = self._apply_probability_threshold(raw_probs)
                v_a = self._calculate_weighted_va(processed_probs)
                res["img"] = {"probs": processed_probs, "va": v_a, "max_idx": np.argmax(raw_probs)}

            # 2. 文本支路推理
            if txt_input is not None:
                ids, mask = [x.to(DEVICE) for x in txt_input]
                logits, _ = self.txt_model(ids, mask)
                raw_probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                processed_probs = self._apply_probability_threshold(raw_probs)
                v_a = self._calculate_weighted_va(processed_probs)
                res["txt"] = {"probs": processed_probs, "va": v_a, "max_idx": np.argmax(raw_probs)}

        # 3. 核心融合逻辑
        if res["img"] and res["txt"]:
            label_img = EMOTIONS[res["img"]["max_idx"]]
            label_txt = EMOTIONS[res["txt"]["max_idx"]]
            w_img, w_txt = 0.5, 0.5

            # --- 步骤 1: 沿用并保留你的 Dominant 阵营判断逻辑 ---
            if label_img == "Neutral" and label_txt == "Neutral":
                w_img, w_txt = 0.6, 0.4
            elif label_img in self.img_dominant and label_txt in self.img_dominant:
                w_img, w_txt = 0.7, 0.3
            elif label_img in self.txt_dominant and label_txt in self.txt_dominant:
                w_img, w_txt = 0.3, 0.7
            # --- 步骤 2: 覆盖逻辑 - 如果出现 Neutral (中性)，则大幅削弱该模态权重 ---
            # 此时会覆盖上面的 w_img/w_txt，确保 Neutral 模态只占 0.1
            if label_img == "Neutral":
                w_img, w_txt = 0.1, 0.9
            elif label_txt == "Neutral":
                w_img, w_txt = 0.9, 0.1

            # 最终合成 VA 坐标
            final_va = w_img * res["img"]["va"] + w_txt * res["txt"]["va"]
            return res, final_va, (w_img, w_txt)

        elif res["img"]:
            return res, res["img"]["va"], (1.0, 0.0)
        elif res["txt"]:
            return res, res["txt"]["va"], (0.0, 1.0)
        return None


class InferenceService:
    """面向 GUI 的推理服务，直接接收路径/文本并返回结构化结果。"""

    def __init__(self):
        self.image_processor = ProcessImage()
        self.text_processor = ProcessText()
        self.engine = FusionEngine(ImageModel(), TextModel(), IMAGE_WEIGHTS, TEXT_WEIGHTS)

    def infer(self, image_path=None, text=None):
        text = (text or "").strip()
        image_tensor = self.image_processor.process_path(image_path) if image_path else None
        text_inputs = self.text_processor.get_processed_input(text) if text else None

        if image_tensor is None and text_inputs is None:
            return None

        output = self.engine.run(image_tensor, text_inputs)
        image_tensor = None
        text_inputs = None
        if not output:
            return None

        details, final_va, weights = output
        return {
            "details": details,
            "final_va": np.asarray(final_va, dtype=float),
            "weights": tuple(float(weight) for weight in weights)
        }

# ================= 4. 模拟运行 =================

def main():
    print("启动双模态情感推理系统...")

    # 初始化
    proc_img = ProcessImage()
    proc_txt = ProcessText()
    engine = FusionEngine(ImageModel(), TextModel(), IMAGE_WEIGHTS, TEXT_WEIGHTS)

    # 模拟读取：一张图 + 一个txt
    img_t = proc_img.get_processed_tensor()

    raw_text = None
    if os.path.exists(INPUT_TEXT_FILE):
        with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()

    # 打印原始文本以便确认
    if raw_text:
        print(f"输入文本: {raw_text}")

    txt_t = proc_txt.get_processed_input(raw_text)

    # 推理
    output = engine.run(img_t, txt_t)

    if output:
        details, final_va, weights = output

        print("\n" + "=" * 45)

        # --- 1. 图像模型结果展示 (全概率) ---
        if details["img"]:
            print(f"[图像模型结果]")
            # 遍历打印所有情绪的概率分布
            for i, prob in enumerate(details['img']['probs']):
                marker = "⭐" if i == details['img']['max_idx'] else "  "
                print(f"   {marker} {EMOTIONS[i]:<10}: {prob * 100:>6.2f}%")
            print(f" 单模态 VA: V:{details['img']['va'][0]:.3f}, A:{details['img']['va'][1]:.3f}")

        # --- 2. 文本模型结果展示 (全概率) ---
        if details["txt"]:
            print(f"\n[文本模型结果]")
            # 遍历打印所有情绪的概率分布
            for i, prob in enumerate(details['txt']['probs']):
                marker = "⭐" if i == details['txt']['max_idx'] else "  "
                print(f"   {marker} {EMOTIONS[i]:<10}: {prob * 100:>6.2f}%")
            print(f"单模态 VA: V:{details['txt']['va'][0]:.3f}, A:{details['txt']['va'][1]:.3f}")

        print("\n" + "=" * 45)

        # --- 3. 最终融合决策展示 ---
        print(f"[融合权重分配]: 图像 {weights[0]:.2f} : 文本 {weights[1]:.2f}")
        print(f"[最终融合坐标]: Valence:{final_va[0]:.4f}, Arousal:{final_va[1]:.4f}")

        # 增加一个象限解析，演示起来更直观
        v, a = final_va[0], final_va[1]
        if v > 0.1:
            state = "积极 (Positive)"
        elif v < -0.1:
            state = "消极 (Negative)"
        else:
            state = "中性 (Neutral)"
        print(f"综合情感倾向: {state}")

        print("=" * 45)
    else:
        print("未检测到有效的输入数据（图片或文本）。")


if __name__ == "__main__":
    main()
