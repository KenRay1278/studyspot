from streamlit.testing.v1 import AppTest


def test_streamlit_app_generates_recommendations():
    app = AppTest.from_file("app.py")
    app.run(timeout=20)
    assert not app.exception
    assert len(app.button) == 1

    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 5
    assert all("/ 5" in metric.value for metric in app.metric)

