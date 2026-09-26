### YuNet

- Modelo: `face_detection_yunet_2023mar.onnx` (232589 bytes), OpenCV Zoo.
- Revisão: `f12e12798e8314f7c074a6656816c048dcc95b7a`.
- Origem: https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet
- Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Licença MIT em `LICENSE-YuNet.txt`.

O arquivo acompanha o projeto. Não há download nem envio de imagens pela rede durante a detecção local.

### SFace

- Modelo: `face_recognition_sface_2021dec.onnx` (38696353 bytes, aproximadamente 37 MiB), OpenCV Zoo.
- Revisão: `f12e12798e8314f7c074a6656816c048dcc95b7a`.
- Origem: https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface
- Download: https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
- SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` (conferido com o ponteiro Git LFS oficial).
- Licença Apache 2.0 preservada em `LICENSE-SFace.txt`.
- Fonte da licença: https://raw.githubusercontent.com/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_recognition_sface/LICENSE
- SHA-256 da licença: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

Reconhecimento inteiramente local, sem download em runtime. `FaceIdentityMatcher` valida os dois modelos no primeiro `match`, alinha pelos cinco landmarks YuNet e compara embeddings SFace normalizados por cosseno. Os padrões conservadores são limiar 0,5 e margem 0,1 entre rótulos distintos; não constituem garantia de identidade. Recortes pequenos, sem rosto único ou com landmarks inadequados não são identificados. Referências ausentes ou inválidas são ignoradas, portanto `new` indica apenas ausência de correspondência visual utilizável. Não há adaptação automática de templates nem comparação por descrição.

### Revisão e unificação de pessoas com biometria facial aprimorada

No módulo `face_identity_review.py` e nas interfaces (`main.py review-people`, `name-people` ou painel gráfico):
- **Extração Biométrica Adaptativa**: O extrator `FaceIdentityMatcher` utiliza padding de contexto adaptativo (`BORDER_REFLECT_101`) para recuperar enquadramentos justos e extrair landmarks faciais e embeddings SFace com alta precisão geométrica.
- **Detecção de Co-ocorrência em Fotos**: Analisa o histórico de fotos cadastradas. Caso duas pessoas apareçam simultaneamente no mesmo enquadramento, o sistema alerta o usuário ou permite filtrar o par, prevenindo falsas unificações de pessoas distintas que foram fotografadas juntas.
- **Classificação em Níveis de Confiança**:
  - `🟢 Alta (≥ 0.65)`: Forte probabilidade de ser a mesma pessoa com variações naturais de iluminação ou ângulo.
  - `🟡 Média (0.45 - 0.65)`: Semelhança visual considerável para inspeção manual.
  - `⚪ Possível (0.35 - 0.45)`: Sugestão para desempate ou casos de baixa resolução.
- **Interfaces GUI e CLI**: Suporte a filtragem dinâmica por limiar, visualização de recortes lado a lado, contagem de fotos associadas e unificação segura mantendo o histórico de imagens. A similaridade é uma sugestão de apoio e nenhum cadastro é mesclado sem confirmação do usuário.

### Fast Local Image Classifier (Fotos vs Prints vs Ícones)

- **Módulo**: `image_classifier.py` (`FastImageClassifier`).
- **Execução**: 100% local, embutida e integrada via OpenCV (`cv2`) e `numpy`, sem necessidade de GPU ou serviços externos (como Ollama).
- **Desempenho**: **~0.5 ms a 2 ms por imagem** em CPU (> 1.000 imagens/segundo).
- **Pipeline em Cascata**:
  1. *Filtro de Metadados & EXIF*: Identificação imediata de câmeras digitais/smartphones (ISO, FNumber, Modelo de Lente) para fotos reais e softwares de captura para prints.
  2. *Análise de Transparência e Geometria*: Detecção de canal alfa, paletas compactas e dimensões quadradas para ícones/assets.
  3. *Análise de Visão Computacional*: Avaliação de alinhamento ortogonal de bordas Sobel (UI vs mundo real), textura Laplaciana, quantização de cores 12-bit e densidade de tiras de texto horizontal.
  4. *Classificador Bayesiano/Softmax Calibrado*: Produção determinística de probabilidades e justificativas entre `photo`, `screenshot`, `icon_or_graphic` e `other`.