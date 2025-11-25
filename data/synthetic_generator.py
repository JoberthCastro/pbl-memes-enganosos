import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from datetime import datetime

# --- Configurações e Constantes ---

OUTPUT_DIR_RAW = os.path.join("data", "raw")
DIRS = {
    "authentic": os.path.join(OUTPUT_DIR_RAW, "authentic"),
    "manipulated": os.path.join(OUTPUT_DIR_RAW, "manipulated")
}

# Textos simulados para conteúdo
TEXTS_AUTHENTIC = [
    "O lançamento do novo foguete foi um sucesso absoluto! 🚀 #Science",
    "Café da manhã reforçado para começar bem o dia. Bom dia a todos!",
    "A economia global mostra sinais de recuperação leve este trimestre.",
    "Vocês viram o jogo ontem? Que virada histórica! ⚽",
    "Estudando Python e amando cada linha de código. #DevLife",
    "Alguém tem indicação de série para maratonar no fim de semana?",
    "A vista daqui de cima é simplesmente espetacular. #Nature",
    "Parabéns a toda a equipe pelo esforço e dedicação no projeto."
]

TEXTS_FAKE_NEWS = [
    "URGENTE: Cientistas confirmam que a terra é oca e habitada!",
    "Governo vai taxar o ar que respiramos a partir de 2025. Compartilhe!",
    "Beba água com limão para curar qualquer doença instantaneamente.",
    "NASA admite que nunca fomos à Lua em documento vazado.",
    "Nova lei proíbe o uso de celulares em locais públicos.",
    "Promoção imperdível: iPhone 15 por apenas R$ 100,00! Clique aqui.",
    "Atenção: O WhatsApp vai passar a ser pago amanhã!"
]

# Lista de ícones (simulados como caracteres ou formas simples)
ICONS = ["♥", "★", "●", "♦", "Like", "Share", "Retweet"]

class SyntheticGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._setup_dirs()
        
        # Tenta carregar fontes, fallback para default
        try:
            self.font_bold = ImageFont.truetype("arialbd.ttf", 18)
            self.font_reg = ImageFont.truetype("arial.ttf", 16)
            self.font_small = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            self.font_bold = ImageFont.load_default()
            self.font_reg = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def _setup_dirs(self):
        for d in DIRS.values():
            os.makedirs(d, exist_ok=True)

    def _draw_tweet_template(self, text, author="User", handle="@user", likes=0, retweets=0):
        """Gera uma imagem simulando um tweet."""
        width, height = 500, 250
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Avatar
        draw.ellipse((20, 20, 70, 70), fill=(200, 200, 200))
        
        # Nome e Handle
        draw.text((80, 25), author, fill="black", font=self.font_bold)
        draw.text((80, 45), handle, fill="gray", font=self.font_small)
        
        # Texto do corpo
        lines = [text[i:i+45] for i in range(0, len(text), 45)]
        y_text = 90
        for line in lines:
            draw.text((20, y_text), line, fill="black", font=self.font_reg)
            y_text += 25
            
        # Data
        draw.text((20, y_text + 10), datetime.now().strftime("%H:%M · %d %b %Y"), fill="gray", font=self.font_small)
        
        # Linha divisória
        y_metrics = y_text + 40
        draw.line((20, y_metrics, width-20, y_metrics), fill=(230, 230, 230), width=1)
        
        # Métricas
        metrics_text = f"{likes} Retweets   {retweets} Likes"
        draw.text((20, y_metrics + 10), metrics_text, fill="black", font=self.font_bold)
        
        return img

    def _draw_whatsapp_template(self, text, time="12:00"):
        """Gera uma imagem simulando mensagem de WhatsApp."""
        width, height = 400, 150
        bg_color = (236, 229, 221) # Cor de fundo padrão WA
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Balão da mensagem (simples)
        bubble_color = (220, 248, 198) # Verde claro
        margin = 20
        
        # Estimativa tamanho texto
        bbox = draw.textbbox((0, 0), text, font=self.font_reg)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        
        bubble_rect = [margin, margin, margin + w_text + 40, margin + h_text + 30]
        draw.rectangle(bubble_rect, fill=bubble_color, outline=(200, 200, 200))
        
        # Texto
        draw.text((margin + 10, margin + 10), text, fill="black", font=self.font_reg)
        
        # Hora
        draw.text((bubble_rect[2] - 40, bubble_rect[3] - 15), time, fill="gray", font=self.font_small)
        
        return img

    def generate_authentic(self, index):
        """Gera um sample autêntico."""
        template_type = random.choice(['tweet', 'whatsapp'])
        text = random.choice(TEXTS_AUTHENTIC)
        
        if template_type == 'tweet':
            img = self._draw_tweet_template(
                text, 
                author=f"User_{index}", 
                handle=f"@user_{index}", 
                likes=random.randint(10, 5000), 
                retweets=random.randint(5, 2000)
            )
        else:
            img = self._draw_whatsapp_template(text)
            
        filename = f"auth_{index:05d}.jpg"
        path = os.path.join(DIRS["authentic"], filename)
        img.save(path)
        
        return filename, "authentic", "none", text

    def apply_manipulation(self, img, manipulation_type):
        """Aplica distorções na imagem."""
        img = img.convert("RGB")
        w, h = img.size
        
        if manipulation_type == "text_swap":
            # Sobrepõe uma caixa branca e escreve texto fake
            draw = ImageDraw.Draw(img)
            # Heurística: cobrir o meio da imagem
            fake_text = random.choice(TEXTS_FAKE_NEWS)
            
            # Simula uma edição mal feita (caixa branca visível)
            box = (20, h//2 - 20, w-20, h//2 + 40)
            draw.rectangle(box, fill="white") # Fundo branco "colado"
            
            # Escreve texto novo (talvez fonte diferente)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            draw.text((25, h//2 - 10), fake_text[:40], fill="black", font=font)
            
        elif manipulation_type == "metrics_change":
            # Altera número de likes grosseiramente
            draw = ImageDraw.Draw(img)
            # Assume que metrics estão na parte inferior (tweet)
            box = (20, h - 40, 150, h - 10)
            draw.rectangle(box, fill="white")
            draw.text((25, h - 35), "999M Likes", fill="black", font=self.font_bold)

        elif manipulation_type == "icon_insertion":
            # Cola um ícone aleatório
            draw = ImageDraw.Draw(img)
            icon = random.choice(ICONS)
            # Posição aleatória
            x = random.randint(0, w-30)
            y = random.randint(0, h-30)
            draw.text((x, y), icon, fill="red", font=self.font_bold)

        elif manipulation_type == "copy_paste":
            # Recorta uma parte e cola em outra (splicing)
            # Converte para array numpy (OpenCV)
            np_img = np.array(img)
            # Crop random region
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            cw, ch = 50, 50
            crop = np_img[y1:y1+ch, x1:x1+cw].copy()
            
            # Paste somewhere else
            x2, y2 = random.randint(w//2, w-cw), random.randint(h//2, h-ch)
            np_img[y2:y2+ch, x2:x2+cw] = crop
            
            img = Image.fromarray(np_img)

        elif manipulation_type == "visual_quality":
            # Contraste/Saturação
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.5, 1.5))
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.0, 2.0))

        elif manipulation_type == "jpeg_compression":
            # Simula salvando e carregando com baixa qualidade
            # Isso é feito no salvamento final, mas podemos forçar artefatos aqui
            pass # Tratado no salvamento

        return img

    def generate_manipulated(self, index):
        """Gera um sample manipulado derivado de um autêntico ou criado do zero."""
        # Cria base autêntica primeiro
        text = random.choice(TEXTS_AUTHENTIC) # Texto original
        img = self._draw_tweet_template(text, likes=100) # Base limpa
        
        manipulation = random.choice([
            "text_swap", "metrics_change", "icon_insertion", 
            "copy_paste", "visual_quality", "jpeg_compression"
        ])
        
        img = self.apply_manipulation(img, manipulation)
        
        filename = f"manip_{index:05d}.jpg"
        path = os.path.join(DIRS["manipulated"], filename)
        
        # Salva com qualidade variável
        quality = 100
        if manipulation == "jpeg_compression":
            quality = random.randint(5, 30) # Qualidade muito baixa
        elif random.random() > 0.7:
            quality = random.randint(50, 80) # Degradação comum
            
        img.save(path, "JPEG", quality=quality)
        
        return filename, "manipulated", manipulation, text

    def run(self, n_authentic=10, n_manipulated=10):
        metadata = []
        print(f"Gerando {n_authentic} imagens autênticas...")
        for i in range(n_authentic):
            meta = self.generate_authentic(i)
            metadata.append(meta)
            
        print(f"Gerando {n_manipulated} imagens manipuladas...")
        for i in range(n_manipulated):
            meta = self.generate_manipulated(i)
            metadata.append(meta)
            
        # Salvar CSV
        df = pd.DataFrame(metadata, columns=["filename", "label", "manipulation_type", "original_text_content"])
        csv_path = os.path.join("data", "labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metadados salvos em {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Gerador de Dados Sintéticos para Memes Enganosos")
    parser.add_argument("--n_authentic", type=int, default=20, help="Número de imagens autênticas")
    parser.add_argument("--n_manipulated", type=int, default=20, help="Número de imagens manipuladas")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    gen = SyntheticGenerator(seed=args.seed)
    gen.run(n_authentic=args.n_authentic, n_manipulated=args.n_manipulated)

if __name__ == "__main__":
    main()
