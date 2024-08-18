import os, pickle, shutil, json, datetime, time, random, string
from typing import List, Any, Tuple
from collections.abc import Callable



def time_checker(start:float) -> str:
    """start(float: time.time())부터 time_checker() 코드 실행까지 걸린 시간을 깔끔하게 출력
    Example: '0:01:55.60'

    Args:
        start (float): 소수초

    Returns:
        str: "시:분:초.밀리초" 형식의 문자열
    """
    # 소모 시간 측정
    end = time.time()
    second_delta = (end - start)
    result = decimal_seconds_to_time_string(decimal_s=second_delta)
    
    return result


def decimal_seconds_to_time_string(decimal_s:float)->str:
    """소수 초 단위 시간을 받아 "시:분:초.밀리초" 형식의 문자열로 변환

    Args:
        decimal_s (_type_): 소수초

    Returns:
        str: "시:분:초.밀리초" 형식의 문자열
    """
    time_delta = datetime.timedelta(seconds=decimal_s)
    str_time_delta = str(time_delta).split(".")
    time1 = str_time_delta[0]
    if len(str_time_delta) == 1:
        time2 = "00"
    else:
        time2 = str_time_delta[1][:2]
    return f"{time1}.{time2}"


def make_random_date_str_id()->str:
    """YYMMDD_HHMMSSff_[0-9a-zA-Z]{size} 라는 중복되기 어려운 임의의 키 생성

    Returns:
        str: 중복이 어려운 임의의 키 생성
    """

    str_key = random_str_id(size=(10,13))



def random_str_id(size:Tuple[int, int]=(10, 20))->str:
    """무작위 길이의 무작위 문자열([0-9a-zA-Z-])을 생성한다.

    Args:
        size (Tuple[int, int], optional): 무작위 문자열의 최대 길이. Defaults to (10, 20).

    Returns:
        str: 무작위 문자열
    """
    possible_chars = string.ascii_letters + string.digits + '-'
    random_size = random.randint(*size)
    return ''.join(random.choices(possible_chars, k=random_size))


def now_datetime_string(date_format="%Y%m%d", time_format="%H%M%S%f")->str:
    """현재 시간을 문자열 key로 생성

    Returns:
        str: f"{date_key}_{time_key}"
    """
    date_key = datetime.datetime.now().strftime(date_format)[2:]
    time_key = datetime.datetime.now().strftime(time_format)[:8]
    return f"{date_key}_{time_key}"


def do_or_load(savepath:str, fn:Callable, makes_new=False, *args, **kwargs)->Any:
    """makes_new가 True거나 filepath에 fn의 결과로 인해 생성되는 파일이 없다면, fn을 동작하고 파일을 저장한다.
    만약 있다면, 그냥 파일을 가져온다.

    Args:
        filepath (str): fn의 결과로 인해 생성되는 데이터가 pickle로 저장될 경로
        fn (Callable): 임의의 함수
        makes_new (bool, optional): filepath에 결과가 있더라도 fn을 동작함. Defaults to False.

    Returns:
        Any: fn의 결과
    """
    if (os.path.exists(savepath) == False) | (makes_new == True):
        result = fn(*args, **kwargs)
        save_pickle(data=result, pickle_path=savepath)
    else:
        result = load_pickle(pickle_path=savepath)
    return result


def new_dir_maker(dir_path:str, makes_new=True):
    """dir_path 디렉터리 생성

    Args:
        dir_path (str): 디렉터리 경로
        makes_new (bool, optional): 디렉터리가 이미 존재하는 경우, 새로 만들지 여부. Defaults to True.
    """
    if os.path.exists(dir_path):
        if makes_new:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            pass
    else:
        os.mkdir(dir_path)


def make_null_list_pickle(pickle_path:str, make_new:bool=False):
    """빈 list가 들어 있는 pickle 파일을 생성한다.

    Args:
        pickle_path (str): pickle 파일의 경로
    """
    
    if (os.path.exists(pickle_path) == False) | (make_new == True):
        data = []
        save_pickle(data=data, pickle_path=pickle_path)


def save_pickle(data:Any, pickle_path:str):
    """data를 pickle_path 경로에 pickle로 저장

    Args:
        data (Any): 대상 데이터
        pickle_path (str): pickle의 경로
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        
def load_pickle(pickle_path:str)->Any:
    """pickle_path 경로의 pickle을 불러옴

    Args:
        pickle_path (str): pickle 파일의 경로

    Returns:
        Any: pickle 안 데이터
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def write_json(data:dict, json_path:str):
    """dictionary를 json_path에 저장
    >>> json에 입력 불가능한 python dtype이 있는 경우, 저장이 안됨

    Args:
        data (dict): dicationary
        json_path (str): 저장할 json 파일의 경로
    """
    with open(json_path, 'w') as f:
        json.dump(data, f)

    
def read_json(json_path:str)->dict:
    """json_path에서 dictionary를 읽어온다.

    Args:
        json_path (str): json 파일의 경로

    Returns:
        dict: json 파일 안에 있던 dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


