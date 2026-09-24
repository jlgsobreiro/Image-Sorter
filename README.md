# ImageSorter

Sistema em Python para:
1. **Varredura e Indexação Recursiva**: Encontra todas as imagens de um disco ou diretório e salva o caminho completo no banco de dados SQLite.
2. **Avaliação Visual com Ollama**: Processa cada imagem registrada no SQLite utilizando modelos de visão multimodal locais do Ollama e salva as descrições no banco de dados.

---

## 📋 Pré-requisitos

- Python 3.10+ (não requer bibliotecas externas pesadas, utiliza bibliotecas padrão do Python).
- [Ollama](https://ollama.com/) instalado e em execução (para a etapa de descrição das imagens).
  - Um modelo com suporte a visão baixado no Ollama, por exemplo:
    ```bash
    ollama run llama3.2-vision
    # ou
    ollama run llava
    # ou
    ollama run moondream
    # ou
    ollama run minicpm-v
    ```

---

## 🗄️ Estrutura do Banco de Dados SQLite

O arquivo padrão gerado é `images.db` com a seguinte tabela:

```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER,
    status TEXT DEFAULT 'pending',  -- 'pending', 'processed', 'error', 'skipped'
    description TEXT,               -- Descrição detalhada incluindo pessoas, cenário e objetos
    people_present TEXT,            -- Lista em JSON de pessoas identificadas (ex: '["Pessoa 1", "Pessoa 2"]')
    model_used TEXT,
    error_message TEXT,
    image_hash TEXT,                -- Hash SHA-256 ou perceptual para detecção de duplicatas
    is_duplicate INTEGER DEFAULT 0, -- 0 = única/principal, 1 = duplicata confirmada
    image_type TEXT DEFAULT 'unclassified', -- 'photo', 'screenshot', 'icon_or_graphic', 'other', 'unclassified'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Tabela de catálogo para consistência de identidade entre imagens:
CREATE TABLE known_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_label TEXT UNIQUE NOT NULL, -- ex: 'Pessoa 1' ou 'João'
    description TEXT NOT NULL,         -- Características fisionômicas duradouras (rosto, idade, traços distintivos)
    first_seen_image_id INTEGER,       -- ID da primeira imagem onde foi registrada
    face_crop_path TEXT,               -- Caminho do recorte contendo apenas o rosto da pessoa
    face_bbox TEXT,                    -- Coordenadas do rosto JSON [ymin, xmin, ymax, xmax]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (first_seen_image_id) REFERENCES images(id)
);
```

### 👥 Consistência de Identidade entre Imagens ("Pessoa 1", "Pessoa 2", ...)

O sistema gerencia um catálogo incremental de indivíduos no SQLite:
1. **Primeira Aparição**: Ao encontrar uma pessoa pela primeira vez, o modelo registra seus traços físicos e fisionômicos duradouros (formato do rosto, tom de pele, cabelo, idade aparente, marcas distintivas) e atribui um identificador sequencial (ex: `Pessoa 1`).
2. **Imagens Subsequentes**: O catálogo de perfis já conhecidos é enviado dinamicamente no prompt do Ollama. O modelo compara as pessoas da nova foto com o catálogo e garante que a **Pessoa 1** seja sempre a mesma pessoa em qualquer imagem em que aparecer.
3. **Novas Pessoas**: Qualquer indivíduo não registrado anteriormente é adicionado ao catálogo como `Pessoa 2`, `Pessoa 3`, etc., enriquecendo a base para as próximas análises.

### 🖼️ Nomeação Interativa e Unificação Facial (`name-people`)

Caso uma pessoa pareça diferente (mudança de ângulo, barba, óculos, iluminação) e o modelo a classifique como outra pessoa (ex: `Pessoa 2`), ou você deseje atribuir nomes reais (ex: `Carlos`, `Mariana`), o ImageSorter disponibiliza uma ferramenta interativa:
- **Exibição do Rosto**: Isola e exibe **o recorte do rosto** da pessoa com recuperação e geração sob demanda resiliente.
- **Renomeação Direta**: Permite atribuir um nome real (ex: renomear `Pessoa 1` para `João`). Todas as imagens onde `Pessoa 1` aparece são atualizadas automaticamente.
- **Unificação / Fusão (Merge)**: Se `Pessoa 2` for a mesma pessoa que `Pessoa 1`, você pode unificá-las com 1 clique. O sistema migra todas as fotos de `Pessoa 2` para `Pessoa 1` e remove o perfil duplicado do catálogo.
- **Acesso ao Visualizador de Fotos**: Permite abrir diretamente as fotos associadas à pessoa no visualizador interativo.
- **Interface Gráfica e Terminal**: Disponível via janela gráfica (GUI) com Tkinter ou via modo terminal (`--cli`).

### 📸 Visualizador de Fotos e Identificação Manual (`view` / `viewer`)

Para inspecionar as fotos cadastradas no acervo e verificar/atribuir os nomes das pessoas diretamente sobre a foto original:
- **Exibição Completa com Bounding Boxes**: Mostra a foto em alta qualidade com retângulos delimitadores coloridos destacando os rostos identificados.
- **🔄 Rotação Interativa da Foto**: Botões e atalhos de teclado para girar a foto em `⟲ -90°`, `⟳ +90°` e `🔄 180°` (atalhos `[R]`, `[L]`, `[Ctrl+R]`, `[Ctrl+L]`), atualizando o arquivo em disco e os metadados no banco.
- **✂️ Recorte e Seleção Manual de Rosto (Crop Forçado)**: Arraste o mouse sobre qualquer região ou rosto na imagem para criar um recorte manual forçado, visualizar a miniatura e atribuir ou cadastrar o nome da pessoa imediatamente.
- **Cards de Identificação na Barra Lateral**: Lista cada rosto detectado com sua respectiva miniatura e uma caixa de seleção editável para escolher uma pessoa do catálogo ou digitar um novo nome.
- **Nomeação Global ou Local**: Permite salvar a alteração apenas para a foto atual ou renomear a pessoa em todo o catálogo global do banco.
- **Detecção Facial Sob Demanda**: Botão para executar a detecção biométrica facial na foto aberta a qualquer momento.
- **Navegação Rápida por Teclado**: Use as setas `[←]` e `[→]` para navegar entre as imagens, `[R]`/`[L]` para girar, `[F5]` para reexecutar a detecção e `[Esc]` para fechar.
- **Edição da Descrição**: Permite ajustar o texto descritivo da imagem diretamente na interface.
- **Modo Terminal e Gráfico**: Execute com interface gráfica ou no modo interativo pelo terminal (`--cli`).

### 🔍 Consultando Pessoas no SQLite

Como a coluna `people_present` armazena a lista de identificadores das pessoas em formato JSON (ex: `["Pessoa 1", "Pessoa 2"]`), você pode realizar consultas diretas:

```sql
-- Buscar todas as fotos onde a "Pessoa 1" está presente:
SELECT id, file_name, file_path, people_present 
FROM images 
WHERE people_present LIKE '%"Pessoa 1"%';

-- Utilizando as funções JSON nativas do SQLite:
SELECT images.id, images.file_name, json_each.value AS pessoa
FROM images, json_each(images.people_present)
WHERE json_each.value = 'Pessoa 1';

-- Listar o catálogo de pessoas identificadas e suas características:
SELECT person_label, description, first_seen_image_id, created_at 
FROM known_people 
ORDER BY id ASC;
```

### ⚡ Reconhecimento Facial Híbrido: Biometria Local (YuNet + SFace) & Descrição Visual (Ollama Vision)

Para maximizar a precisão, eliminar alucinações e acelerar o processamento:
1. **Detecção e Alinhamento Facial (OpenCV YuNet)**:
   - Rede neural leve (`models/face_detection_yunet_2023mar.onnx`) que roda na CPU e localiza com precisão os rostos e 5 pontos fiduciais (olhos, nariz e cantos da boca).
   - Examina fotos em múltiplas escalas com supressão de não-máximos (NMS), aceitando rostos em diversos ângulos e resoluções.
2. **Comparação Biométrica Visual de Alta Precisão (OpenCV SFace)**:
   - Em vez de depender de descrições textuais subjetivas ou prompts do LLM para decidir se duas pessoas são a mesma, o sistema utiliza o modelo de extração de características faciais **SFace** (`models/face_recognition_sface_2021dec.onnx`).
   - O SFace alinha a face usando os marcos do YuNet, extrai um vetor de embedding de 128 dimensões e calcula a similaridade cosseno contra o catálogo de referências (`known_people`).
   - Possui limiar conservador (`threshold=0.5`) e margem de desambiguação (`margin=0.1`). Se a pessoa já for conhecida, a correspondência é imediata e determinística na CPU, dispensando chamadas redundantes ao LLM.
   - Mantém cache seguro em memória baseado no estado do arquivo de recorte (`face_crop_path`).
3. **Descrição Geral da Foto com IA (Ollama Vision) [Opcional]**:
   - O modelo multimodal do Ollama (`llava`, `llama3.2-vision`, `moondream`, etc.) pode ser invocado para descrever a foto como um todo: cenário, ambiente, iluminação, objetos e ações.
   - A chamada à IA é **totalmente opcional**: se você deseja apenas catalogar pessoas e indexar o acervo de fotos rapidamente via biometria local, basta utilizar a flag `--no-ai` (ou desmarcar a opção na interface gráfica), economizando tempo e recursos computacionais.
   - Não são feitas chamadas redundantes de IA para recortes individuais de rostos: a identificação e agrupamento de pessoas são sempre resolvidos com máxima eficiência e precisão na CPU pela biometria local.
4. **Economia Computacional e Salto de Imagens sem Rostos**:
   - Imagens sem rostos detectados têm a inferência pesada dispensada automaticamente, acelerando o fluxo de processamento de grandes acervos.
   - O registro é marcado no SQLite como processado (`status = 'processed'`, `people_present = '[]'`) e o sistema passa instantaneamente para a próxima imagem.
   - A flag `--no-face-detect` pode ser usada caso deseje desativar a detecção local e avaliar toda e qualquer imagem diretamente com o modelo de visão.
   - Falhas de leitura ou de todos os detectores são registradas como erro, não como ausência de rostos, permitindo tentar novamente.

**Imagens já processadas:** a melhoria vale para as próximas avaliações; o banco existente não é modificado automaticamente. Imagens anteriormente marcadas com `model_used = 'face_detection_skip'` precisam voltar ao status `pending` caso você queira reavaliá-las. Faça backup do banco antes de qualquer alteração manual.

**Testes:** `python -m unittest test_face_detection test_imagesorter` inclui inferência real em uma fotografia pública da NASA, imagens vazias, coordenadas multiescala, fallback e tratamento de falhas. Não é uma medição de precisão sobre seu acervo.

### 🖼️ Classificação com IA (Fotos Reais vs Prints vs Ícones) & Exportação de Fotos

Para separar fotografias autênticas de capturas de tela e gráficos de software:
1. **Classificação Multimodal Especializada (Ollama Vision)**:
   - Categoriza cada imagem em uma das classes:
     - `photo` (📷 **Foto Real**): retratos, viagens, fotos de família, paisagens, animais e objetos do mundo real.
     - `screenshot` (📱 **Print de Tela**): capturas de celular, conversas de WhatsApp/Telegram, janelas de programas, navegadores web.
     - `icon_or_graphic` (🎨 **Ícone / Asset de Software**): botões de interface, logotipos, clipart, ilustrações vetoriais, diagramas.
     - `other` (📄 **Outro / Documento**): documentos escaneados, memes com texto predominante, texturas.
   - Análise rápida e determinística com modelo local (`llava`, `moondream`, `llama3.2-vision`).
2. **Exportação Exclusiva de Fotos Reais para Pasta**:
   - Cria uma pasta contendo **exclusivamente as fotos reais** (`photo`), descartando automaticamente prints de tela e ícones de software.
   - Integração com o seletor nativo de pastas do Windows Explorer.
   - Permite filtrar para exportar apenas fotos únicas (eliminando duplicatas repetidas).
3. **Filtros no Visualizador de Fotos**:
   - O visualizador de fotos (`python main.py view`) permite filtrar a exibição por `📷 Apenas Fotos Reais`, `📱 Apenas Prints de Tela` ou `🎨 Apenas Ícones/Assets`.

### 👯 Detecção de Duplicatas & Exportação de Imagens Únicas

Para otimizar o armazenamento em disco e evitar fotos redundantes:
1. **Identificação por Hash (SHA-256 e Perceptual dHash)**:
   - O sistema calcula o fingerprint digital de cada imagem cadastrada.
   - Agrupa instantaneamente arquivos que possuem exatamente o mesmo conteúdo visual ou binário.
2. **Comparador Visual de Duplicatas com Ciclagem**:
   - Interface em quadros lado a lado exibindo todas as fotos do grupo duplicado, tamanho de arquivo, dimensões e pessoas identificadas.
   - Permite ciclar pelos grupos (`Grupo Anterior`, `Próximo Grupo`), abrir a imagem em tela cheia e escolher com 1 clique qual imagem manter como principal (`⭐ Manter esta`).
3. **Exportação de Imagens Únicas com Seletor do Windows Explorer**:
   - Cria uma nova pasta com **apenas uma cópia de cada foto única** (descartando todas as duplicatas).
   - Abre a caixa de diálogo nativa do Windows Explorer para seleção da pasta de destino e abre a pasta finalizada ao concluir.

---

## 🎮 Modos Interativos Completos (CLI e GUI)

Todas as funcionalidades do ImageSorter agora possuem suporte **interativo completo**:

### 1. Menu Interativo Principal no Terminal (CLI)
Ao executar `main.py` sem argumentos ou com a flag `-i`:
```bash
python main.py
# ou
python main.py interactive
# ou
python main.py -i
```
O sistema abre um painel interativo no terminal permitindo:
- 📂 **Varredura Interativa (Scan)**: Pergunta o diretório/disco (detecta unidades como `C:\`, `D:\`), extensões e lote com validação imediata.
- 🤖 **Avaliação Interativa com IA (Describe)**: Conecta ao Ollama, lista automaticamente os modelos de visão instalados para seleção numérica, permite definir limite de imagens e personalizar prompts.
- 👤 **Identificação & Nomeação Facial**: Escolha entre abrir a Janela Gráfica (GUI) ou Modo Terminal (CLI) para ver o rosto recortado e renomear/unificar identidades.
- 🔍 **Explorador & Busca de Imagens/Pessoas**: Lista todas as pessoas catalogadas, fotos onde aparecem, permite buscar por texto e **abrir fotos no visualizador padrão do Windows/SO** diretamente.
- ⚡ **Pipeline Completo Interativo**: Guia o fluxo ponta a ponta (Varredura -> Avaliação IA -> Nomeação de Pessoas).
- 📊 **Status e Diagnósticos do Banco de Dados**.

### 2. Avaliador Visual Interativo com IA ao Vivo (3 Quadros Sincronizados)
Para acompanhar exatamente o que a inteligência artificial e a visão computacional estão fazendo em tempo real:
```bash
python main.py live
# ou
python main.py describe --gui
```
A tela é dividida em **3 quadros principais**:
- **Quadro 1 (Esquerda - Imagem em Avaliação)**: Exibe a foto atual sendo avaliada, redimensionada proporcionalmente, com caixas delimitadoras coloridas desenhadas sobre cada rosto detectado e status em tempo real.
- **Quadro 2 (Direita - Rostos Detectados)**: Galeria com miniaturas dos recortes faciais isolados em memória, identificação atribuída (`Pessoa 1`, `João`), badge indicando perfil novo/conhecido e resumo de traços biométricos.
- **Quadro 3 (Inferior - Texto da IA & Descrição)**: Exibe as pessoas presentes identificadas e a descrição textual completa gerada pelo Ollama em tempo real com tempos de resposta.
- **Controles de Fluxo**:
  - 🔄 **Modo Contínuo**: Processa automaticamente imagem por imagem.
  - ⏸️ **Modo Passo a Passo**: Processa a imagem atual e pausa para você inspecionar com calma antes de avançar para a próxima.
  - Botões de Pausar, Retomar, Próxima Imagem e Interromper.

### 3. Painel Gráfico Integrado Geral (GUI Dashboard Tkinter)
Para controle de todas as funções (Varredura, Processamento IA, Nomeação de Pessoas, Duplicatas e Busca):
```bash
python main.py --gui
# ou
python main.py gui
```
- **Barra de Progresso e Percentual em Tempo Real**: Exibe o andamento detalhado e contagem de imagens em operações de varredura (Scan), recálculo de hashes e exportação.
- **Botão de Interromper**: Permite cancelar/pausar qualquer operação em andamento com 1 clique (`⏹️ Interromper Operação`), liberando os recursos do sistema com segurança.
- **Execução Assíncrona Não-Bloqueante**: Todas as tarefas longas rodam em segundo plano mantendo a interface gráfica 100% responsiva.

### 4. Modos Interativos nos Scripts Individuais
- **Varredura Interativa**: `python scan_images.py -i`
- **Avaliação Interativa com Ollama**: `python describe_images.py -i`
- **Avaliador Visual ao Vivo (3 Quadros)**: `python main.py live`
- **Nomeação de Pessoas**: `python main.py name-people` (GUI) ou `python main.py name-people --cli` (CLI)
- **Explorador / Busca**: `python main.py explore`

---

## 🚀 Como Usar via Linha de Comando (Modo Direto / Automação)

Você pode executar os scripts individualmente ou através do ponto de entrada unificado `main.py`.

### 1. Escanear imagens de um disco ou pasta (`scan_images.py`)

Para escanear todo o disco `C:\` ou um diretório específico:

```bash
# Escaneando uma pasta específica
python scan_images.py "C:\Users\Nome\Pictures"

# Escaneando um disco inteiro
python scan_images.py "D:\"

# Definindo banco de dados personalizado e extensões específicas
python scan_images.py "C:\Fotos" --db minhas_imagens.db --ext jpg png webp
```

### 2. Avaliar imagens com o Ollama (`describe_images.py`)

Certifique-se de que o Ollama está rodando (`ollama serve`):

```bash
# Executa a avaliação com o modelo padrão (llava)
python describe_images.py

# Executa apenas reconhecimento facial e catalogação biométrica local (sem IA / sem Ollama)
python describe_images.py --no-ai

# Usando outro modelo (ex: llama3.2-vision) e limitando a 50 imagens por execução
python describe_images.py --model llama3.2-vision --limit 50

# Customizando o prompt de análise
python describe_images.py --prompt "Identifique se há pessoas nesta foto e liste os objetos visíveis em português."

# Retentando imagens que falharam anteriormente
python describe_images.py --retry-errors
```

### 3. Utilizando o `main.py` (Interface Unificada)

```bash
# Ver status do banco de dados
python main.py status

# Executar varredura
python main.py scan "C:\Fotos"

# Executar descrições
python main.py describe --model llama3.2-vision

# Subcomando: pipeline (scan + describe)
python main.py pipeline "C:\Fotos" --model llama3.2-vision

# Nomear pessoas interativamente com interface gráfica (GUI com visualização do rosto)
python main.py name-people

# Nomear pessoas de forma interativa diretamente no terminal (CLI)
python main.py name-people --cli

# Abrir o visualizador interativo de fotos para ver fotos e nomear pessoas na imagem (GUI)
python main.py view

# Abrir visualizador de fotos filtrando por uma pessoa específica
python main.py view --person "Carlos"

# Visualizador de fotos diretamente pelo terminal (CLI)
python main.py view --cli

# Identificar duplicatas e abrir o comparador visual de quadros (GUI)
python main.py duplicates

# Revisar duplicatas diretamente no terminal (CLI)
python main.py duplicates --cli

# Exportar apenas imagens únicas para uma pasta (abre seletor do Windows Explorer)
python main.py export-unique

# Exportar imagens únicas para pasta específica via linha de comando
python main.py export-unique "C:\Fotos_Unicas"

# Classificar imagens em Fotos Reais, Prints e Ícones com IA (Ollama)
python main.py classify --model llava

# Exportar exclusivamente Fotos Reais para uma pasta (abre seletor do Windows Explorer)
python main.py export-photos

# Exportar Fotos Reais para uma pasta específica via linha de comando
python main.py export-photos "C:\Minhas_Fotos_Reais"

# Ativar logs detalhados (DEBUG) e salvar em arquivo
python main.py --debug --log-file execucao.log scan "C:\Fotos"
```

---

## 📜 Logs em Tempo Real

Todos os comandos agora possuem saída de logs detalhada com data/hora, nível e contexto de cada operação:
- **Varredura (`scan`)**: Registra diretórios analisados, imagens encontradas, lotes salvos no SQLite e resumo estatístico final.
- **Avaliação (`describe`)**: Registra progresso individual `[x/total]`, codificação em Base64, requisições enviadas ao Ollama, tempo de resposta por imagem, prévia da descrição gerada e atualização no banco SQLite.
- **Classificação (`classify`)**: Identifica o tipo de cada imagem com IA (Foto Real, Print de Tela, Ícone/Asset de software) e exibe diagnósticos e confiança.
- **Exportação (`export-photos` / `export-unique`)**: Rastreia a cópia segura de arquivos, resolução de colisões e relatório de economia em megabytes.
- **Parâmetros opcionais de log**:
  - `--debug`: Exibe informações aprofundadas de depuração em tempo real.
  - `--log-file caminho.log`: Salva todo o histórico de execução em um arquivo texto.

---

## 🧪 Executando os Testes

Para rodar a suíte completa de testes unitários automatizados:

```bash
python -m unittest discover -v
```
