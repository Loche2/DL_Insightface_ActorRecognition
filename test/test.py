result = [
    [{'idx': 0, 'name': 'amy', 'sim': 0.01828331}, {'idx': 1, 'name': 'amy', 'sim': 0.075875506}, {'idx': 2, 'name': 'amy', 'sim': 0.46815157}, {'idx': 3, 'name': 'amy', 'sim': 0.03415847}],
    [{'idx': 0, 'name': 'sheldon', 'sim': 0.03686998}, {'idx': 1, 'name': 'sheldon', 'sim': 0.54803985}, {'idx': 2, 'name': 'sheldon', 'sim': 0.07452112},
     {'idx': 3, 'name': 'sheldon', 'sim': -0.008363429}]]
sorted_data = sorted(result, key=lambda x: x['sim'], reverse=True)
print(sorted_data)
