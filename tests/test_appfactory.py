from tests import BaseTestCase
from spamfilter.models import db, File


class TestAppFactory(BaseTestCase):

    def test_config(self):
        self.assertTrue(self.app.testing)

    def test_appfactory_home(self):
        response = self.client.get('/home')
        self.assertIn(response.data, b'This verifies Application working Status : test')
        self.assertEqual(self.app, self.app)  # <Flask 'myflaskproject'>
        self.assertEqual(self.app, self.app)

