__version__ = '1.0.1'

__version_descriptions__ = {  
    '1.0.0':  [  
        '2023-12-14',  
        ],  
    '1.0.1':  [  
        '2024-03-02',
        'duplicate data issue resolved in BO initialization'
        ],  
}

print(f'pyBO version: {__version__}. updated on {__version_descriptions__[__version__][0]}')


from .pyBO import *
