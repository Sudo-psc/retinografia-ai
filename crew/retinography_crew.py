from crewai import Agent, Task, Crew, Process
from crewai_tools import VisionTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisMetrics:
    """Métricas da análise de retinografia"""
    confidence_score: float
    processing_time: float
    quality_score: float
    validation_score: float

@dataclass
class AnalysisResult:
    """Resultado detalhado da análise de retinografia"""
    abnormalities: List[Dict[str, Any]]
    features: Dict[str, Any]
    metrics: AnalysisMetrics
    recommendations: List[str]
    diagnostic_confidence: float
    validation_notes: List[str]

    def __init__(self, abnormalities=None, features=None):
        self.abnormalities = abnormalities or []
        self.features = features or {
            "optic_disc": {
                "location": None,
                "diameter": None,
                "condition": None,
                "confidence": 0.0
            },
            "macula": {
                "location": None,
                "condition": None,
                "confidence": 0.0
            },
            "vessels": {
                "detected": False,
                "pattern": None,
                "abnormalities": [],
                "confidence": 0.0
            }
        }
        self.metrics = AnalysisMetrics(
            confidence_score=0.0,
            processing_time=0.0,
            quality_score=0.0,
            validation_score=0.0
        )
        self.recommendations = []
        self.diagnostic_confidence = 0.0
        self.validation_notes = []

class ImageValidator:
    """Validador de imagens de retinografia"""
    
    @staticmethod
    def validate_image(image_path: str) -> tuple[bool, str]:
        """Valida a qualidade e adequação da imagem"""
        try:
            if isinstance(image_path, str):
                img = Image.open(image_path)
            else:
                img = image_path
                
            # Validar dimensões mínimas
            if img.size[0] < 500 or img.size[1] < 500:
                return False, "Imagem muito pequena para análise adequada"
            
            # Validar formato de cor
            if img.mode not in ['RGB', 'L']:
                return False, "Formato de cor não suportado"
            
            # Validar contraste e brilho
            img_array = np.array(img)
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            if brightness < 20 or brightness > 235:
                return False, "Brilho da imagem fora do intervalo adequado"
            if contrast < 10:
                return False, "Contraste da imagem muito baixo"
                
            return True, "Imagem válida para análise"
            
        except Exception as e:
            logger.error(f"Erro na validação da imagem: {str(e)}")
            return False, f"Erro na validação: {str(e)}"

class RetinographyCrewManager:
    """Gerenciador do sistema de análise de retinografia"""
    
    def __init__(self):
        self.vision_tool = VisionTool()
        self.openai = ChatOpenAI(model="gpt-4o")
        self.anthropic = ChatAnthropic(model="claude-3.5-sonnet-20241020")
        self.image_validator = ImageValidator()

    def create_agents(self):
        """Cria agentes especializados para análise"""
        
        # Agente Analista de Imagem
        image_analyst = Agent(
            role="Retinal Image Analyst",
            goal="Analisar imagens retinianas e detectar características com alta precisão",
            backstory="""Especialista em análise de imagens retinianas com profundo 
                        conhecimento em processamento de imagens médicas, padrões 
                        patológicos e biomarcadores retinianos. Possui experiência 
                        em identificação de anomalias sutis e variações anatômicas.""",
            llm=self.openai,
            tools=[
                self.vision_tool,
                "image_enhancement",
                "feature_detection",
                "vessel_analysis"
            ],
            verbose=True
        )

        # Agente Especialista Médico
        medical_specialist = Agent(
            role="Ophthalmology Specialist",
            goal="Fornecer interpretação médica precisa dos achados retinianos",
            backstory="""Oftalmologista experiente especializado em doenças da retina,
                        com vasta experiência em diagnóstico por imagem e correlações
                        clínico-patológicas. Possui expertise em identificação precoce
                        de patologias e avaliação de progressão de doenças.""",
            llm=self.anthropic,
            tools=[
                "medical_analysis",
                "pathology_identification",
                "disease_progression",
                "treatment_recommendation"
            ],
            verbose=True
        )

        # Agente Redator de Laudos
        report_writer = Agent(
            role="Medical Report Writer",
            goal="Gerar laudos médicos abrangentes e precisos",
            backstory="""Especialista em documentação médica com expertise em
                        terminologia oftalmológica, estruturação de laudos e
                        comunicação médica efetiva. Possui habilidade em sintetizar
                        informações complexas em relatórios claros e objetivos.""",
            llm=self.anthropic,
            tools=[
                "report_generation",
                "terminology_validation",
                "report_structuring",
                "recommendation_formatting"
            ],
            verbose=True
        )

        # Agente Validador
        validator = Agent(
            role="Quality Assurance Specialist",
            goal="Validar e garantir a qualidade das análises",
            backstory="""Especialista em controle de qualidade com foco em
                        validação de análises médicas e garantia de precisão
                        diagnóstica. Possui experiência em identificação de
                        inconsistências e validação cruzada de resultados.""",
            llm=self.anthropic,
            tools=[
                "quality_validation",
                "cross_reference",
                "consistency_check",
                "confidence_assessment"
            ],
            verbose=True
        )

        return [image_analyst, medical_specialist, report_writer, validator]

    def create_tasks(self, image_path: str) -> List[Task]:
        """Cria tarefas para análise da retinografia"""
        agents = self.create_agents()
        
        tasks = [
            Task(
                description=f"""Analise detalhada da imagem retiniana em {image_path}:
                1. Pré-processamento e validação da imagem
                2. Detecção e análise do disco óptico:
                   - Localização e dimensões
                   - Medida da relação escavação disco
                   - Medida do tamanho do disco (pequeno,médio ou grande)
                   - Características morfológicas
                   - Presença de sinais de atrofia peripapilar
                   - Avaliação do cotorno e coloração
                   - Alterações patológicas
                3. Avaliação da mácula:
                   - Localização e integridade
                   - Características do brilho foveal
                   - Presença ou ausências de drusas, edema ou hemorragias
                   - Alterações estruturais
                   - Sinais de patologia
                4. Mapeamento vascular:
                   - Padrão de distribuição
                   - Alterações calibrosas
                   - Anomalias vasculares, tortuosidades e cruzamentos patológicos
                   - Sinais de hemorragias
                5. Identificação de lesões e alterações:
                   - Hemorragias
                   - Exsudatos
                   - Drusas
                   - Alterações pigmentares""",
                agent=agents[0]
            ),
            Task(
                description="""Interpretação médica abrangente:
                1. Avaliação das características detectadas:
                   - Significado clínico
                   - Correlações patológicas
                2. Identificação de condições patológicas:
                   - Diagnósticos diferenciais
                   - Estágio da doença
                3. Avaliação de progressão:
                   - Indicadores de atividade
                   - Fatores de risco
                4. Correlações clínicas:
                   - Impacto funcional
                   - Prognóstico
                5. Recomendações terapêuticas:
                   - Opções de tratamento
                   - Monitoramento""",
                agent=agents[1]
            ),
            Task(
                description="""Geração de laudo médico estruturado:
                1. Compilação dos achados:
                   - Dados técnicos
                   - Achados normais e patológicos
                2. Estruturação profissional:
                   - Formato padronizado
                   - Seções organizadas
                3. Terminologia médica precisa:
                   - Nomenclatura atual
                   - Classificações padronizadas
                4. Conclusões e recomendações:
                   - Síntese diagnóstica
                   - Plano de ação
                5. Documentação complementar:
                   - Imagens relevantes
                   - Medidas e métricas""",
                agent=agents[2]
            ),
            Task(
                description="""Validação e controle de qualidade:
                1. Verificação de consistência:
                   - Entre análises dos agentes
                   - Com padrões estabelecidos
                2. Avaliação de confiança:
                   - Métricas de qualidade
                   - Níveis de certeza
                3. Validação cruzada:
                   - Correlação entre achados
                   - Confirmação de diagnósticos
                4. Garantia de qualidade:
                   - Completude do laudo
                   - Precisão das informações""",
                agent=agents[3]
            )
        ]
        
        return tasks

    def analyze_retinography(self, image_path: str) -> AnalysisResult:
        """Executa o workflow completo de análise de retinografia"""
        try:
            # Validar imagem
            valid, message = self.image_validator.validate_image(image_path)
            if not valid:
                logger.error(f"Validação da imagem falhou: {message}")
                return AnalysisResult()

            # Criar e executar tarefas
            tasks = self.create_tasks(image_path)
            crew = Crew(
                agents=self.create_agents(),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Executar análise
            result = crew.kickoff()
            
            # Processar e estruturar resultado
            processed_result = self._process_crew_result(result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Erro na análise de retinografia: {str(e)}")
            return AnalysisResult()

    def _process_crew_result(self, crew_result: Any) -> AnalysisResult:
        """Processa e estrutura o resultado da análise"""
        try:
            # Criar resultado base
            result = AnalysisResult()
            
            # Extrair e estruturar informações do resultado
            if hasattr(crew_result, 'final_output'):
                # Processar anormalidades
                if 'abnormalities' in crew_result.final_output:
                    result.abnormalities = crew_result.final_output['abnormalities']
                
                # Processar características
                if 'features' in crew_result.final_output:
                    result.features.update(crew_result.final_output['features'])
                
                # Processar recomendações
                if 'recommendations' in crew_result.final_output:
                    result.recommendations = crew_result.final_output['recommendations']
                
                # Processar métricas
                if 'metrics' in crew_result.final_output:
                    result.metrics = AnalysisMetrics(**crew_result.final_output['metrics'])
                
                # Processar confiança diagnóstica
                if 'diagnostic_confidence' in crew_result.final_output:
                    result.diagnostic_confidence = crew_result.final_output['diagnostic_confidence']
                
                # Processar notas de validação
                if 'validation_notes' in crew_result.final_output:
                    result.validation_notes = crew_result.final_output['validation_notes']
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento do resultado: {str(e)}")
            return AnalysisResult()
