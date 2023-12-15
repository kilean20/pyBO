__version__ = '1.0.0'

__version_descriptions__ = {  
    '1.0.0':  [  
        '2023-12-14',  
        ],  
}

print(f'pyBO version: {__version__}. updated on {__version_descriptions__[__version__][0]}')


from .pyBO import *
