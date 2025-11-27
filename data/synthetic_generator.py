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

# Textos simulados para conteúdo (Baseado em notícias reais e neutras)
TEXTS_AUTHENTIC = [
    "O novo relatório do IPCC alerta para o aumento da temperatura global nos próximos anos. #Clima",
    "Governo anuncia redução de 5% no imposto para empresas de tecnologia verde.",
    "A campanha de vacinação contra a gripe começa na próxima segunda-feira em todo o país.",
    "Brasil vence a Argentina por 3x0 e garante vaga na final do campeonato. ⚽",
    "Receita Federal libera a consulta ao terceiro lote de restituição do Imposto de Renda.",
    "Estudo da USP mostra que prática regular de exercícios reduz riscos cardíacos em 30%.",
    "A taxa de desemprego caiu para 8,5% no último trimestre, segundo dados do IBGE.",
    "Lançamento do novo satélite vai melhorar a previsão do tempo no Nordeste.",
    "Festival de cinema premia diretor brasileiro com o Leão de Ouro. 🎬",
    "Trânsito intenso na Avenida Paulista devido a obras de manutenção no asfalto.",
    "Ministério da Saúde reforça importância de manter carteira de vacinação atualizada.",
    "Nova atualização do sistema bancário permitirá transferências instantâneas internacionais.",
    "Cientistas descobrem nova espécie de orquídea na Mata Atlântica.",
    "Bolsa de Valores fecha em alta de 1,2% puxada pelo setor de commodities.",
    "Prefeitura inaugura 5 novas creches na zona leste da cidade hoje.",
    "O consumo de energia elétrica aumentou 4% em relação ao mesmo período do ano passado.",
    "Começam hoje as inscrições para o concurso público com 500 vagas.",
    "Dólar opera em baixa nesta terça-feira, cotado a R$ 4,95.",
    "Museu de Arte Moderna abre exposição gratuita sobre o Modernismo no Brasil.",
    "Pesquisa aponta crescimento do comércio eletrônico no primeiro semestre."
]

# Textos simulados para conteúdo enganoso (Baseado em fake news comuns: política, saúde, conspiração)
TEXTS_FAKE_NEWS = [
    "URGENTE: Ministro confirma que vai confiscar a poupança de todos os brasileiros em 2025!",
    "Médico de Harvard revela: 'Beber água gelada com limão cura câncer em 3 dias'. Compartilhe!",
    "Vaza áudio onde o candidato X admite que vai acabar com o Bolsa Família se for eleito.",
    "ONU aprova resolução que obriga escolas a ensinarem 'ideologia de gênero' a bebês.",
    "Cientistas admitem em segredo que o Aquecimento Global é uma farsa para vender carros elétricos.",
    "Documento vazado da NASA prova que a Terra é plana e o governo esconde a borda no Polo Sul.",
    "STF decide secretamente que é crime cantar o Hino Nacional e usar a bandeira do Brasil.",
    "Atenção: O WhatsApp será cobrado a partir de amanhã! Mande para 10 pessoas para evitar.",
    "Vacinas contêm microchips líquidos para rastrear a população, afirma ex-funcionário da CIA.",
    "Foto comprova que o ex-presidente foi visto jantando com líder de facção criminosa ontem.",
    "Bancos vão bloquear o CPF de quem não atualizar os dados cadastrais até hoje à meia-noite.",
    "Nova lei de trânsito: Multa de R$ 3.000 para quem dirigir de chinelo a partir de sábado.",
    "China cria 'sol artificial' para controlar o clima mundial e causar secas no Ocidente.",
    "Hospital esconde a cura do diabetes para lucrar com a venda de insulina. Veja a receita natural!",
    "Urnas eletrônicas foram programadas para transferir 20% dos votos para o candidato da oposição.",
    "Governo vai distribuir 'kit gay' nas creches a partir do mês que vem. Absurdo!",
    "Bilionário George Soros está financiando invasão alienígena para instaurar a Nova Ordem Mundial.",
    "Decreto secreto proíbe o consumo de carne vermelha no país a partir de 2030.",
    "Beber urina pela manhã aumenta a imunidade e previne todas as doenças virais, diz especialista.",
    "Fim da propriedade privada? Nova lei propõe que o governo pode tomar sua casa se tiver quarto sobrando."
]

# Lista de ícones (simulados como caracteres ou formas simples)
ICONS = ["♥", "★", "●", "♦", "Like", "Share", "Retweet", "⚠", "❌", "fake"]

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
            self.font_large = ImageFont.truetype("arial.ttf", 20) # Fonte maior para manchetes
        except IOError:
            self.font_bold = ImageFont.load_default()
            self.font_reg = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def _setup_dirs(self):
        for d in DIRS.values():
            os.makedirs(d, exist_ok=True)

    def _draw_tweet_template(self, text, author="User", handle="@user", likes=0, retweets=0):
        """Gera uma imagem simulando um tweet."""
        width, height = 600, 300 # Aumentei um pouco
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Avatar
        draw.ellipse((20, 20, 70, 70), fill=(200, 200, 200))
        
        # Nome e Handle
        draw.text((80, 25), author, fill="black", font=self.font_bold)
        draw.text((80, 45), handle, fill="gray", font=self.font_small)
        
        # Texto do corpo (Quebra de linha manual simples)
        max_chars = 50
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y_text = 90
        for line in lines:
            draw.text((20, y_text), line, fill="black", font=self.font_large) # Fonte maior no corpo
            y_text += 30
            
        # Data
        draw.text((20, y_text + 15), datetime.now().strftime("%H:%M · %d %b %Y"), fill="gray", font=self.font_small)
        
        # Linha divisória
        y_metrics = y_text + 45
        draw.line((20, y_metrics, width-20, y_metrics), fill=(230, 230, 230), width=1)
        
        # Métricas
        metrics_text = f"{likes} Retweets   {retweets} Likes"
        draw.text((20, y_metrics + 10), metrics_text, fill="black", font=self.font_bold)
        
        return img

    def _draw_whatsapp_template(self, text, time="12:00"):
        """Gera uma imagem simulando mensagem de WhatsApp."""
        width, height = 500, 250
        bg_color = (236, 229, 221) # Cor de fundo padrão WA
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Balão da mensagem (simples)
        bubble_color = (220, 248, 198) # Verde claro
        margin = 20
        
        # Estimativa tamanho texto (simplificada)
        # Quebra de linha
        max_chars = 45
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Calcula altura do balão baseado nas linhas
        line_height = 20
        h_text = len(lines) * line_height
        w_text = 300 # Largura fixa aproximada para simplificar
        
        bubble_rect = [margin, margin, margin + w_text + 40, margin + h_text + 30]
        draw.rectangle(bubble_rect, fill=bubble_color, outline=(200, 200, 200))
        
        # Texto
        y_line = margin + 10
        for line in lines:
            draw.text((margin + 10, y_line), line, fill="black", font=self.font_reg)
            y_line += line_height
        
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
                author=f"Jornal_Real_{index}", 
                handle=f"@jornal_real_{index}", 
                likes=random.randint(100, 5000), 
                retweets=random.randint(50, 2000)
            )
        else:
            img = self._draw_whatsapp_template(text)
            
        filename = f"auth_{index:05d}.jpg"
        path = os.path.join(DIRS["authentic"], filename)
        
        # MODIFICADO: Salva com qualidade aleatória também, para confundir a rede neural
        # e forçá-la a não usar compressão como feature discriminativa.
        quality = random.randint(60, 95)
        img.save(path, "JPEG", quality=quality)
        
        return filename, "authentic", "none", text

    def apply_manipulation(self, img, manipulation_type):
        """Aplica distorções na imagem."""
        img = img.convert("RGB")
        w, h = img.size
        
        if manipulation_type == "text_swap":
            # Sobrepõe uma caixa branca e escreve texto fake
            draw = ImageDraw.Draw(img)
            fake_text = random.choice(TEXTS_FAKE_NEWS)
            
            # Heurística: tenta cobrir a área do texto original
            # No tweet, texto começa em y=90. No WA, margem=20.
            # Vamos cobrir uma área central generica
            box = (20, 80, w-20, h-50) # Cobre quase todo o corpo
            draw.rectangle(box, fill="white") 
            
            # Escreve texto novo
            max_chars = 45
            lines = []
            words = fake_text.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            y_text = 90
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            for line in lines:
                draw.text((25, y_text), line, fill="black", font=font)
                y_text += 25
            
        elif manipulation_type == "metrics_change":
            # Altera número de likes grosseiramente
            draw = ImageDraw.Draw(img)
            # Assume que metrics estão na parte inferior (tweet)
            box = (20, h - 40, 200, h - 10)
            draw.rectangle(box, fill="white")
            fake_metrics = f"{random.randint(100,900)}K Retweets  {random.randint(10,900)}M Likes"
            draw.text((25, h - 35), fake_metrics, fill="black", font=self.font_bold)

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
            np_img = np.array(img)
            # Crop random region
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            cw, ch = 60, 60
            crop = np_img[y1:y1+ch, x1:x1+cw].copy()
            
            # Paste somewhere else
            x2, y2 = random.randint(w//2, w-cw), random.randint(h//2, h-ch)
            np_img[y2:y2+ch, x2:x2+cw] = crop
            
            img = Image.fromarray(np_img)

        elif manipulation_type == "visual_quality":
            # Contraste/Saturação
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.5, 2.0))
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.0, 2.5))

        elif manipulation_type == "jpeg_compression":
            pass # Tratado no salvamento

        return img

    def generate_manipulated(self, index):
        """Gera um sample manipulado derivado de um autêntico ou criado do zero."""
        # Cria base autêntica primeiro
        text_base = random.choice(TEXTS_AUTHENTIC) 
        img = self._draw_tweet_template(text_base, likes=random.randint(10,500)) 
        
        manipulation = random.choice([
            "text_swap", "metrics_change", "icon_insertion", 
            "copy_paste", "visual_quality", "jpeg_compression"
        ])
        
        img = self.apply_manipulation(img, manipulation)
        
        # Se a manipulação foi trocar o texto, o texto final na imagem mudou
        # Para outros, o texto ainda é o original
        final_text = text_base
        if manipulation == "text_swap":
            # Como apply_manipulation escolhe aleatoriamente, é difícil saber qual foi escolhido.
            # Idealmente, deveríamos refatorar para passar o texto.
            # Mas para simplificar, vamos assumir que o label do CSV vai refletir "manipulated"
            # e o modelo vai aprender que texto != imagem ou texto suspeito
            pass

        filename = f"manip_{index:05d}.jpg"
        path = os.path.join(DIRS["manipulated"], filename)
        
        # MODIFICADO: Aumenta a qualidade mínima para que não seja tão óbvio
        # Antes: quality = random.randint(5, 20) se compression
        # Agora: quality = random.randint(40, 90) para ficar mais parecido com as autênticas
        quality = random.randint(60, 95) # Default range parecido com autênticas
        
        if manipulation == "jpeg_compression":
             # Ainda degrada, mas menos brutalmente, ou degrada autênticas também
             # Vamos manter a degradação como característica, mas menos extrema
            quality = random.randint(30, 60) 
        
        img.save(path, "JPEG", quality=quality)
        
        return filename, "manipulated", manipulation, final_text

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
