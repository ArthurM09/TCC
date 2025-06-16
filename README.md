# Segmentação e Classificação de ARIA Landmarks em Aplicações Web

📌 **Repositório do TCC**  
Código-fonte e dados do Trabalho de Conclusão de Curso de **Arthur Massuia Miranda**  
Curso: Engenharia de Computação  
Universidade: UTFPR (Universidade Tecnológica Federal do Paraná)  

---

## 📜 Sobre o Projeto
A acessibilidade digital é fundamental para garantir que pessoas com deficiência possam navegar na web de forma autônoma. Os **ARIA Landmarks** (como `<nav>`, `<main>`, `role="banner"`) são componentes essenciais que estruturam páginas web, permitindo que tecnologias assistivas ofereçam navegação direta e compreensível.

### Desafio e Objetivo
A implementação correta desses landmarks ainda é um desafio para desenvolvedores. Este projeto explora técnicas de **Aprendizado de Máquina Supervisionado** para automatizar a identificação e classificação de ARIA landmarks, visando criar ferramentas que contribuam para uma web mais acessível.

### Algoritmos Avaliados
1. Árvore de Decisão  
2. Support Vector Machine (SVM)  
3. Multi-Layer Perceptron (MLP)  

---

## 🚀 Como Executar o Projeto
Siga os passos para replicar o ambiente, coletar dados, treinar modelos e gerar análises.

### 1. Preparando o Ambiente
**Pré-requisitos:**
- Python 3.8+
- Docker instalado

**Instalar dependências Python:**
```bash
pip install selenium pandas numpy scikit-learn matplotlib seaborn
# Atualizar pip (opcional/recomendado):
python -m pip install --upgrade pip
