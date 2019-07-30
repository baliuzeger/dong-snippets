import json
import dong.framework

class DefaultService(dong.framework.Service):

    @dong.framework.request_handler
    def hello(self, request_body, mime_type='application/json'): 

        return json.dumps('hello')
