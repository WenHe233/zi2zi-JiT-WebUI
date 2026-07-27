from util.compile import compile_if_available, torch_compile_available


def test_torch_compile_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("ZI2ZI_TORCH_COMPILE", "off")

    def identity(value):
        return value

    assert not torch_compile_available()
    assert compile_if_available(identity) is identity
