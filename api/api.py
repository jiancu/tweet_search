#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function

from pathlib import Path
import falcon
import spacy
import json

import falcon

class QuoteResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': (
                "I've always been more interested in "
                "the future than in the past."
            ),
            'author': 'Grace Hopper'
        }

        resp.media = quote

api = falcon.API()
api.add_route('/quote', QuoteResource())