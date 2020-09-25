# this file does not have anything to do with the gan
# this is an attempt to disable file downloand in jupyter

from tornado import web
from notebook.base.handlers import IPythonHandler

class ForbidFilesHandler(IPythonHandler):
  @web.authenticated
  def head(self, path):
    self.log.info("HEAD: File download forbidden.")
    #raise web.HTTPError(403)
    return None 

  @web.authenticated
  def get(self, path, include_body=True):
    self.log.info("GET: File download forbidden.")
    #raise web.HTTPError(403)
    return None 
