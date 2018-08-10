import datetime
import json
import requests

APP_KEY = '814c59b24b1da8eca1db0b0e28a04c1c'

def time_converter(time):
    converted_time = datetime.datetime.fromtimestamp(
        int(time)
    ).strftime('%Y-%m-%d %A %I:%M %p')
    return converted_time

def url_builder_name(city_name):

    unit = 'metric'
    api = 'http://api.openweathermap.org/data/2.5/weather?q='

    full_api_url = api + str(city_name) + '&lang=zh_cn' + '&units=' + unit + '&APPID=' + APP_KEY
    return  full_api_url

def data_fetch(full_api_url):
    respone = requests.get(full_api_url)

    try:
        respone.raise_for_status()
    except Exception as exc:
        print('There was a problem:{}'.format(exc))

    return json.loads(respone.text)

def data_organizer(raw_data):
    main = raw_data.get('main')
    sys = raw_data.get('sys')
    data = {
        'city': raw_data.get('name'),
        'country': sys.get('country'),
        'temp': main.get('temp'),
        'temp_max': main.get('temp_max'),
        'temp_min': main.get('temp_min'),
        'humidity': main.get('humidity'),
        'pressure': main.get('pressure'),
        'sky': raw_data['weather'][0]['main'],
        'sunrise': time_converter(sys.get('sunrise')),
        'sunset': time_converter(sys.get('sunset')),
        'wind': raw_data.get('wind').get('speed'),
        'wind_deg': raw_data.get('wind').get('deg'),
        'dt': time_converter(raw_data.get('dt')),
        'cloudiness': raw_data.get('clouds').get('all'),
        'description': raw_data['weather'][0]['description']
    }

    return data

def data_output(data):

    data['m_symbol'] = '\u00b0' + 'C'


    s = '''
--------------------------------------------------------------------
    Current weather in : {city}, {country}:
    {temp}{m_symbol} {sky}
    最高气温: {temp_max}{m_symbol}, 最低气温: {temp_min}{m_symbol}

    风速: {wind}米/秒, 风向: {wind_deg}
    湿度: {humidity} %
    云量: {cloudiness} %
    气压: {pressure} hPa
    日出时间: {sunrise}
    日落时间: {sunset}
    Description: {description}

    Last update from the server: {dt}
-------------------------------------------------------------------------'''

    print(s.format(**data))



city = input("Which city you want to check:")

url = url_builder_name(city)
rawData = data_fetch(url)
prettyData = data_organizer(rawData)
data_output(prettyData)























