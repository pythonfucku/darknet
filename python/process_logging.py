#!/bin/env python3
import sys
import logging
import logging.config

#DATA_TYPE = '%Y%m%d%H%M%S'
DATA_TYPE = "%Y-%m-%d %H:%M:%S"

FORMAT_STR = '%(asctime)s [%(process)d] %(levelname)s > %(message)s'
#FORMAT_STR = '%(asctime)s,%(process)d,%(thread)d,%(levelname)s > %(message)s'

COLOR = "process_logging.ColorFormatter"
VERSION = sys.version_info.major

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        logging.Formatter.__init__(self)
        if VERSION>= 3:
            super().__init__(fmt, datefmt, style)
        else:
            super(ColorFormatter,self).__init__(fmt,datefmt)

    def format(self, record):
        if record.levelno == logging.WARNING:
            record.msg = '\x1b[95m%s\x1b[0m' % record.msg
        elif record.levelno == logging.ERROR:
            record.msg = '\x1b[91m%s\x1b[0m' % record.msg
        elif record.levelno == logging.DEBUG:
            record.msg = '\x1b[92m%s\x1b[0m' % record.msg
        return logging.Formatter.format(self, record)


def initLogging(logfile):

    LOGGING_DIC = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': FORMAT_STR,
                'datefmt': DATA_TYPE,
            },
            'colordefault': {
                'format': FORMAT_STR,
                'datefmt': DATA_TYPE,
            },
	    },
        'filters': {},
        'handlers': {
                'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'colordefault'
                },
                'default': {
                    'level': 'DEBUG',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': logfile, 
                    'maxBytes': 1024 * 1024 * 50,
                    'backupCount': 8,
                    'formatter': 'colordefault',
                    'encoding': 'utf-8',
                },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'console', ],
                'level': 'DEBUG',
                'propagate': True,
            }
        },
    }

    if VERSION>=3:
        LOGGING_DIC["formatters"]["colordefault"]["class"] = COLOR
    else:
        LOGGING_DIC["formatters"]["colordefault"]["()"] = COLOR


    logging.config.dictConfig(LOGGING_DIC)
    #logger = logging.getLogger(__name__)




    """
		'error': {
		    'level': 'ERROR',
		    'class': 'logging.handlers.RotatingFileHandler',
		    'filename': ,
		    'maxBytes': 1024 * 1024 * 5, 
		    'backupCount': 5,
		    'formatter': 'standard',
		    'encoding': 'utf-8',
		},
    """




