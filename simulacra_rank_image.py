from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
from simulacra_aesthetic_models.simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from CLIP import clip

class SimulacraAesthetic():
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.clip_model_name = 'ViT-B/16'
        self.clip_model = clip.load(self.clip_model_name, jit=False, device=self.device)[0]
        self.clip_model.eval().requires_grad_(False)

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        self.model = AestheticMeanPredictionLinearModel(512)
        self.model.load_state_dict(
            torch.load("simulacra_aesthetic_models/models/sac_public_2022_06_29_vit_b_16_linear.pth")
        )
        self.model = self.model.to(self.device)

    def predict(self, img):
        img = img.convert('RGB')
        img = TF.resize(img, 224, interpolation=transforms.InterpolationMode.LANCZOS)
        img = TF.center_crop(img, (224,224))
        img = TF.to_tensor(img).to(self.device)
        img = self.normalize(img)
        clip_image_embed = F.normalize(
            self.clip_model.encode_image(img[None, ...]).float(),
            dim=-1)
        score = self.model(clip_image_embed)

        return score