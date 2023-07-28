'''
Provide several methods to log infomation:
    1. Through logging package in python
    2. Through TensorBoard
'''

import logging
import datetime
import json

# reference by: https://github.com/qindongliang/python_log_json/blob/master/format/json_formatter.py

class JsonFormatter(logging.Formatter):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f" #+ ".%03d" % (now.microsecond / 1000) + "Z"
    DATE_STANDA = 'LOCAL'
    ORIGINAL_KEYS = [
        'name', 'msg', 'args', 'levelname', 'levelno', 
        'pathname', 'filename', 'module', 'exc_info', 
        'exc_text', 'stack_info', 'lineno', 'funcName', 
        'created', 'msecs', 'relativeCreated', 'thread', 
        'threadName', 'processName', 'process'
        ]
    
    COMMON_WORDS = {
        "@timestamp", "@level"
    }
    
    
    
    def format(self, record):
        extra = self.build_record(record)
        self.set_format_time(extra)

        formatted = {}
        if isinstance(record.msg, dict):
            
            for key, value in record.msg.items():
                
                if value in self.COMMON_WORDS:
                    formatted[key] = extra.get(value)
                else:
                    formatted[key] = value
        print(formatted)
        # exit()
        # return formatted
        # return json.dumps(formatted, indent=1, ensure_ascii=False)
        return json.dumps(formatted, ensure_ascii=False)
        
    @classmethod
    def set_format_time(cls, extra):
        if cls.DATE_STANDA.lower() == 'local':
            now = datetime.datetime.now()
        else:
            now = datetime.datetime.utcnow()
            
        dateFormat = cls.DATE_FORMAT
        format_time = now.strftime(dateFormat)
        extra['@timestamp'] = format_time
        
        return format_time
    
    @classmethod
    def build_record(cls, record):
        _extracted = {
            attr_name: record.__dict__[attr_name]
            for attr_name in record.__dict__
        }
        
        _extracted["@level"] = _extracted["levelname"]
        return _extracted

if __name__ == "__main__":
    
    l = logging.getLogger()
    
    l.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler()    
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonFormatter())
    
    l.addHandler(handler)
    
    
    test_info = {'logM': "Test logger", "logP": 28, "logC": "hahah", 'logT': "@timestamp", 'logL': "@level"}
    
    
    l.info(test_info)
    