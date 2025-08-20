Programa de teste de uma MLP simples utilizando keras e sklearn

Gerenciador de dependências: 
- Pipenv

Dependências do projeto:
- numpy = "2.3.2"
- pandas = "2.3.1"
- matplotlib = "3.10.5"
- scikit-learn = "1.7.1"
- tqdm = "4.67.1"

Passos Pipenv:
- Abrir terminal
- pipenv install (Build)
Ative o ambiente:
- pipenv shell
- pipenv seu_script.py
Sem ativar o ambiente:
- pipenv run python seu_script.py

Comandos do gerenciador de deps:
- pipenv graph                #ver árvore de dependências
- pipenv update               # atualizar tudo
- pipenv install pacote==x.y  # adicionar mais um
- pipenv uninstall pacote     # remover
- pipenv lock --clear         # refazer o lock do zero
- pipenv sync                 # instalar exatamente o que está no Pipfile.lock
- pipenv --venv               # caminho do venv
- pipenv --rm                 # deletar o venv (reset)


Fork ------------------------------------------------------------------------------------------------------------------------
Uma branch request foi realizada pelo outro usuário gerando conflito, ambos alterando o título de um gráfico gerado pelo MLP.
O conflito foi resolvido pelo dono original do arquivo, commitado e comentado posteriormente. 
