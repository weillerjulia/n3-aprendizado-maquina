import pytest
from streamlit.testing.v1 import AppTest

from src.dados import ENTRADAS, EXCLUIDAS, PERFIL, RAIZ, carregar, separar
from src.previsao import LIMITES, OPCOES, carregar_modelo, prever, validar

CLIENTE = {"age": 35, "job": "technician", "marital": "married",
           "education": "professional.course", "default": "no", "housing": "yes", "loan": "no",
           "contact": "cellular", "campaign": 1, "previous": 0, "poutcome": "nonexistent"}


@pytest.fixture(scope="module")
def dados():
    return carregar()


@pytest.fixture(scope="module")
def pacote():
    return carregar_modelo()


def test_perfis_nao_se_repetem_entre_treino_e_teste(dados):
    treino, teste = separar(dados)
    assert not set(map(tuple, treino[PERFIL].values)) & set(map(tuple, teste[PERFIL].values))
    assert len(treino) + len(teste) == len(dados)


def test_colunas_excluidas_nao_entram_no_modelo(dados):
    assert not set(ENTRADAS) & set(EXCLUIDAS)
    assert set(ENTRADAS) | set(EXCLUIDAS) == set(dados) - {"y", "alvo"}


def test_interface_cobre_todos_os_valores_da_base(dados):
    for campo, opcoes in OPCOES.items():
        assert set(dados[campo]) == set(opcoes)
    for campo, limites in LIMITES.items():
        assert (dados[campo].min(), dados[campo].max()) == limites


def test_toda_linha_da_base_passa_na_validacao(dados):
    assert not any(validar(linha) for linha in dados[ENTRADAS].to_dict("records"))


@pytest.mark.parametrize("campo,valor", [
    ("age", 999), ("age", 16), ("age", 30.5), ("job", "xxx"), ("campaign", 0),
    ("poutcome", "success"), ("previous", 2),
])
def test_entrada_invalida_e_recusada(pacote, campo, valor):
    with pytest.raises(ValueError):
        prever({**CLIENTE, campo: valor}, pacote)


@pytest.mark.parametrize("campo,valor", [("age", 70), ("job", "blue-collar")])
def test_perfil_do_cliente_muda_a_previsao(pacote, campo, valor):
    original = prever(CLIENTE, pacote)["probabilidade"]
    assert prever({**CLIENTE, campo: valor}, pacote)["probabilidade"] != original


def test_aplicativo_com_exemplos_e_entrada_contraditoria():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=90).run()
    vistos = []
    for exemplo in app.selectbox(key="exemplo").options:
        app.selectbox(key="exemplo").set_value(exemplo).run()
        assert not app.exception and not app.error
        vistos.append(app.metric[1].value)
        print(exemplo, [m.value for m in app.metric[:3]])
    assert set(vistos) == {"Alta", "Média", "Baixa"}

    app.number_input(key="previous").set_value(2).run()
    assert app.error and not app.exception
