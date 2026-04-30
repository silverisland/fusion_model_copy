数据存储为Dataframe格式。一下是对DF的字段描述
1. timestamp_win: 时间戳；分辨率为15min
2. observe_power: 历史7天光功率；每个值为长度672的np.darray;
3. observe_power_future:未来2天光功率；每个值长度为192的np.darray; 一般用作target；
4. GHI_solargis: 历史7天GHI；每个值为长度672的np.darray;
5. GHI_solargis_future:未来2天GHI；每个值长度为192的np.darray;
6. TEMP_solargis: 历史7天温度；每个值为长度672的np.darray;
7. TEMP_solargis_future:未来2天温度；每个值长度为192的np.darray;
