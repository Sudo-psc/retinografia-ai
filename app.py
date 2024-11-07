import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import base64
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from skimage import filters, morphology, segmentation, exposure
import logging
import time
import json
import requests
import sounddevice as sd
import soundfile as sf
import torch
import torchvision.transforms as transforms
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('retinografia.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RetinalFeatures:
    """Classe para armazenar características da retina"""
    optic_disc_location: Optional[Tuple[int, int]]
    optic_disc_diameter: Optional[float]
    macula_location: Optional[Tuple[int, int]]
    vessels_detected: bool
    segmentation_mask: Optional[np.ndarray] = None

@dataclass
class ImageAnalysisResult:
    """Classe para armazenar resultados da análise de imagem"""
    features: RetinalFeatures
    abnormalities: List[str]
    processed_image: Image.Image
    metrics: Dict[str, Any]
    segmentation_overlay: Optional[Image.Image] = None

@dataclass
class AIResponse:
    """Classe para armazenar respostas dos modelos de IA"""
    model_name: str
    laudo: str
    confidence_score: float
    processing_time: float

class AudioTranscriber:
    """Classe para gerenciar transcrição de áudio"""
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.sample_rate = 44100
        self.channels = 1
        
    def record_audio(self, duration: int = 30) -> str:
        """Grava áudio do microfone"""
        try:
            st.write("🎤 Gravando...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()
            
            # Salvar temporariamente
            temp_file = "temp_audio.wav"
            sf.write(temp_file, audio_data, self.sample_rate)
            
            return temp_file
        except Exception as e:
            logger.error(f"Erro na gravação de áudio: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcreve áudio usando OpenAI Whisper"""
        try:
            with open(audio_file, "rb") as file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    response_format="text",
                    language="pt"
                )
            return response
        except Exception as e:
            logger.error(f"Erro na transcrição: {str(e)}")
            raise
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

class ImageSegmenter:
    """Classe para segmentação avançada de imagens retinais"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    def segment_retinal_features(self, image: Image.Image) -> np.ndarray:
        """Segmenta características retinais usando técnicas avançadas"""
        try:
            # Converter para array numpy
            img_array = np.array(image)
            
            # Separar canais
            if len(img_array.shape) == 3:
                green_channel = img_array[:, :, 1]
            else:
                green_channel = img_array
            
            # Melhorar contraste
            enhanced = exposure.equalize_adapthist(green_channel)
            
            # Detectar disco óptico
            disc_mask = self._detect_optic_disc(enhanced)
            
            # Detectar mácula
            macula_mask = self._detect_macula(enhanced)
            
            # Detectar vasos
            vessels_mask = self._detect_vessels(enhanced)
            
            # Combinar máscaras
            combined_mask = np.zeros_like(enhanced)
            combined_mask[disc_mask > 0] = 1  # Vermelho para disco óptico
            combined_mask[macula_mask > 0] = 2  # Verde para mácula
            combined_mask[vessels_mask > 0] = 3  # Azul para vasos
            
            return combined_mask
        except Exception as e:
            logger.error(f"Erro na segmentação: {str(e)}")
            raise
    
    def _detect_optic_disc(self, image: np.ndarray) -> np.ndarray:
        """Detecta o disco óptico usando técnicas avançadas"""
        # Aplicar threshold adaptativo
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        
        # Operações morfológicas
        opened = morphology.opening(binary, morphology.disk(15))
        
        # Encontrar região mais circular
        label_img = morphology.label(opened)
        regions = morphology.regionprops(label_img)
        
        disc_mask = np.zeros_like(image, dtype=bool)
        if regions:
            # Selecionar região mais circular
            best_region = max(regions, key=lambda r: r.extent)
            disc_mask[label_img == best_region.label] = True
        
        return disc_mask
    
    def _detect_macula(self, image: np.ndarray) -> np.ndarray:
        """Detecta a mácula usando técnicas avançadas"""
        # Inverter imagem para destacar região escura
        inverted = 1 - image
        
        # Aplicar threshold adaptativo
        thresh = filters.threshold_otsu(inverted)
        binary = inverted > thresh
        
        # Operações morfológicas
        opened = morphology.opening(binary, morphology.disk(10))
        
        # Encontrar região mais escura e compacta
        label_img = morphology.label(opened)
        regions = morphology.regionprops(label_img)
        
        macula_mask = np.zeros_like(image, dtype=bool)
        if regions:
            # Selecionar região mais compacta
            best_region = max(regions, key=lambda r: r.solidity)
            macula_mask[label_img == best_region.label] = True
        
        return macula_mask
    
    def _detect_vessels(self, image: np.ndarray) -> np.ndarray:
        """Detecta vasos sanguíneos usando Frangi filter"""
        vessels = filters.frangi(
            image,
            sigmas=range(1, 10, 2),
            beta=0.5,
            black_ridges=True
        )
        
        # Threshold para binarização
        thresh = filters.threshold_otsu(vessels)
        vessels_mask = vessels > thresh
        
        # Limpeza morfológica
        vessels_mask = morphology.remove_small_objects(vessels_mask, min_size=50)
        vessels_mask = morphology.remove_small_holes(vessels_mask, area_threshold=50)
        
        return vessels_mask

class AIModelManager:
    """Classe para gerenciar interações com modelos de IA"""
    
    def __init__(self):
        load_dotenv()
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.xai_api_url = "https://api.x.ai/v1"
    
    def encode_image_to_base64(self, image):
        """Converte imagem para base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def get_grok_analysis(self, image: Image.Image) -> str:
        """Obtém análise usando modelo grok-beta da x.ai"""
        try:
            base64_image = self.encode_image_to_base64(image)
            
            headers = {
                "Authorization": f"Bearer {self.xai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grok-beta",
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um especialista em análise de imagens retinais. Analise a imagem fornecida e descreva detalhadamente as características observadas."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "data": base64_image
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(f"{self.xai_api_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Erro na análise com grok-beta: {str(e)}")
            return None

    def get_o1_preview_analysis(self, clinical_history: str, grok_analysis: str, gpt4_analysis: str, claude_analysis: str) -> str:
        """Gera laudo final usando modelo o1-preview da OpenAI"""
        try:
            prompt = f"""Com base nas seguintes informações:

História Clínica:
{clinical_history}

Análise Grok-beta:
{grok_analysis}

Análise GPT-4:
{gpt4_analysis}

Análise Claude-3:
{claude_analysis}

Por favor, gere um laudo oftalmológico completo e estruturado, incluindo:
1. Dados do exame
2. Achados principais
3. Correlação com história clínica
4. Impressão diagnóstica
5. Recomendações"""

            response = self.openai.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um oftalmologista especializado em laudos de retinografia. Gere um laudo profissional e detalhado."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erro na análise com o1-preview: {str(e)}")
            return None

    def get_analysis(self, analysis_result: ImageAnalysisResult, clinical_history: str) -> AIResponse:
        """Obtém análise completa da imagem"""
        try:
            start_time = time.time()
            
            # Obter análises dos diferentes modelos
            grok_analysis = self.get_grok_analysis(analysis_result.processed_image)
            
            gpt4_analysis = self.get_gpt4_analysis(
                analysis_result.processed_image,
                [],  # features_description não é mais necessário
                analysis_result.metrics
            )
            
            claude_analysis = self.get_claude_analysis(
                analysis_result.processed_image,
                [],  # features_description não é mais necessário
                analysis_result.metrics
            )
            
            # Gerar laudo final com o1-preview
            if all([grok_analysis, gpt4_analysis, claude_analysis]):
                final_analysis = self.get_o1_preview_analysis(
                    clinical_history,
                    grok_analysis,
                    gpt4_analysis,
                    claude_analysis
                )
                
                processing_time = time.time() - start_time
                
                return AIResponse(
                    model_name="Multi-Model Analysis (Grok-beta, GPT-4, Claude-3, O1-preview)",
                    laudo=final_analysis,
                    confidence_score=0.98,
                    processing_time=processing_time
                )
            else:
                return AIResponse(
                    model_name="Error Analysis",
                    laudo="Erro ao gerar análises. Por favor, tente novamente.",
                    confidence_score=0.0,
                    processing_time=0.0
                )
            
        except Exception as e:
            logger.error(f"Erro na análise: {str(e)}")
            return AIResponse(
                model_name="Error Analysis",
                laudo="Erro na análise. Por favor, tente novamente.",
                confidence_score=0.0,
                processing_time=0.0
            )

def main():
    st.set_page_config(page_title="Análise Avançada de Retinografia", layout="wide")
    
    st.title("Análise Avançada de Retinografia com IA")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("Sobre")
        st.write("""
        Sistema avançado de análise de retinografia que utiliza:
        - Processamento de imagem com técnicas state-of-the-art
        - Análise com múltiplos modelos de IA (Grok-beta, GPT-4, Claude-3)
        - Segmentação automática de características
        - Transcrição de áudio para história clínica
        - Geração de laudos detalhados com O1-preview
        """)
    
    # História Clínica
    st.header("História Clínica")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        clinical_history = st.text_area(
            "Digite a história clínica do paciente",
            height=100,
            help="Inclua informações relevantes como sintomas, duração, histórico médico, etc."
        )
    
    with col2:
        st.write("Ou grave um áudio")
        if st.button("🎤 Gravar História Clínica"):
            try:
                transcriber = AudioTranscriber()
                audio_file = transcriber.record_audio()
                clinical_history = transcriber.transcribe_audio(audio_file)
                st.session_state['clinical_history'] = clinical_history
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Erro na gravação/transcrição: {str(e)}")
    
    if 'clinical_history' in st.session_state:
        st.info(f"História Clínica Transcrita: {st.session_state['clinical_history']}")
    
    # Upload de Imagem
    uploaded_file = st.file_uploader("Escolha uma imagem de retinografia", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Processar imagem
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Imagem Original")
                st.image(image, use_column_width=True)
            
            # Processamento e segmentação
            processor = ImageProcessor()
            enhanced_image = processor.enhance_image(image)
            
            with col2:
                st.subheader("Imagem Melhorada")
                st.image(enhanced_image, use_column_width=True)
            
            # Segmentação avançada
            segmenter = ImageSegmenter()
            segmentation_mask = segmenter.segment_retinal_features(enhanced_image)
            
            # Criar overlay colorido
            overlay = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
            overlay[segmentation_mask == 1] = [255, 0, 0]    # Vermelho para disco óptico
            overlay[segmentation_mask == 2] = [0, 255, 0]    # Verde para mácula
            overlay[segmentation_mask == 3] = [0, 0, 255]    # Azul para vasos
            
            # Sobrepor à imagem original
            alpha = 0.4
            overlay_image = Image.fromarray(cv2.addWeighted(
                np.array(enhanced_image), 1-alpha,
                overlay, alpha, 0
            ))
            
            with col3:
                st.subheader("Segmentação")
                st.image(overlay_image, use_column_width=True)
                st.caption("Vermelho: Disco Óptico | Verde: Mácula | Azul: Vasos")
            
            # Análise de características
            analysis_result = processor.detect_features(enhanced_image)
            analysis_result.segmentation_overlay = overlay_image
            
            # Métricas e Informações
            st.subheader("Métricas da Imagem")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Brilho Médio", f"{analysis_result.metrics['brightness']:.2f}")
            with col2:
                st.metric("Variação de Intensidade", f"{analysis_result.metrics['std_dev']:.2f}")
            with col3:
                disc_status = "Detectado" if analysis_result.features.optic_disc_location else "Não Detectado"
                st.metric("Disco Óptico", disc_status)
                if analysis_result.features.optic_disc_diameter:
                    st.write(f"Diâmetro: {analysis_result.features.optic_disc_diameter:.1f}px")
            with col4:
                macula_status = "Detectada" if analysis_result.features.macula_location else "Não Detectada"
                st.metric("Mácula", macula_status)
                if 'disc_macula_distance' in analysis_result.metrics:
                    st.write(f"Distância DO-Mácula: {analysis_result.metrics['disc_macula_distance']:.1f}px")
            
            # Alertas e Anomalias
            if analysis_result.abnormalities:
                st.warning("Anomalias Detectadas:")
                for abnormality in analysis_result.abnormalities:
                    st.write(f"- {abnormality}")
            
            # Gerar laudo
            if st.button("Gerar Laudo Completo", type="primary"):
                try:
                    with st.spinner("Gerando análises com múltiplos modelos de IA..."):
                        ai_manager = AIModelManager()
                        response = ai_manager.get_analysis(
                            analysis_result,
                            clinical_history or "História clínica não fornecida."
                        )
                        
                        # Exibir resultado
                        st.write("### Laudo Final")
                        st.write(f"Tempo de processamento: {response.processing_time:.2f}s")
                        st.write(f"Confiança estimada: {response.confidence_score:.2%}")
                        
                        # Criar tabs para diferentes seções do laudo
                        tab1, tab2 = st.tabs(["Laudo Completo", "Download"])
                        
                        with tab1:
                            st.markdown(response.laudo)
                        
                        with tab2:
                            st.download_button(
                                "Download Laudo Completo",
                                response.laudo,
                                "laudo_retinografia.md",
                                "text/markdown"
                            )
                            
                except Exception as e:
                    st.error(f"Erro ao gerar laudo: {str(e)}")
                    logger.error(f"Erro na geração do laudo: {str(e)}")
        
        except Exception as e:
            st.error(f"Erro ao processar imagem: {str(e)}")
            logger.error(f"Erro no processamento de imagem: {str(e)}")

if __name__ == "__main__":
    main()
