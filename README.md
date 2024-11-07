# Retinografia AI

Sistema avançado de análise de retinografia que utiliza múltiplos modelos de IA para auxiliar no diagnóstico oftalmológico.

## Funcionalidades

- Processamento de imagem com técnicas state-of-the-art
- Análise com múltiplos modelos de IA (Grok-beta, GPT-4, Claude-3)
- Segmentação automática de características retinianas
- Detecção e análise de disco óptico, mácula e vasos sanguíneos
- Transcrição de áudio para história clínica
- Geração de laudos detalhados com O1-preview

## Requisitos

- Python 3.9+
- OpenAI API Key
- Anthropic API Key
- X.AI API Key

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/retinografia-ai.git
cd retinografia-ai
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
Crie um arquivo `.env` com as seguintes variáveis:
```
OPENAI_API_KEY=sua_chave_aqui
ANTHROPIC_API_KEY=sua_chave_aqui
XAI_API_KEY=sua_chave_aqui
```

## Uso

Execute o aplicativo com:
```bash
streamlit run app.py
```

O aplicativo estará disponível em `http://localhost:8501`

## Funcionalidades Detalhadas

### Processamento de Imagem
- Melhoramento automático de contraste e nitidez
- Redução de ruído adaptativa
- Segmentação avançada de estruturas retinianas

### Análise de IA
- Análise multi-modelo para maior precisão
- Detecção automática de anomalias
- Geração de laudos estruturados

### Interface
- Upload de imagens de retinografia
- Gravação e transcrição de história clínica
- Visualização de métricas e análises em tempo real
- Download de laudos em formato markdown

## Segurança

Este projeto segue as melhores práticas de segurança:
- Não armazena dados sensíveis
- Utiliza variáveis de ambiente para chaves de API
- Implementa validação de entrada
- Logs seguros sem informações sensíveis

## Contribuição

Contribuições são bem-vindas! Por favor, siga estes passos:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter)

Link do Projeto: [https://github.com/seu-usuario/retinografia-ai](https://github.com/seu-usuario/retinografia-ai)
