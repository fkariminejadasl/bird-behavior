import behavior.data as bd


def test_get_data():
    device_id = 534
    start_time = "2012-06-08 10:28:58"
    database_url = "postgresql://username:yourpass@pub.e-ecology.nl:5432/eecology"
    gimus, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
    assert gimus.shape == (60, 4)
