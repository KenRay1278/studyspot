import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("entrypoint", ["app.py", "streamlit_app.py"])
def test_streamlit_app_generates_recommendations(entrypoint):
    app = AppTest.from_file(entrypoint)
    app.run(timeout=20)
    assert not app.exception
    assert len(app.button) == 1

    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 5
    assert all("/ 5" in metric.value for metric in app.metric)
