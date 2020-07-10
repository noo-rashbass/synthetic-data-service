from sqlalchemy import create_engine
from params import params

def connect(username, password):
    p = params.copy()
    p['username'] = username
    p['password'] = password

    db = create_engine('{dialect}+{driver}://{username}:{password}@{server}:{port}/{database}'.format(**p))
    return db
