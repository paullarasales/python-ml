marvel_heroes = ['spider man', 'thor', 'hulk', 'iron man', 'captian america']

print(len(marvel_heroes))

marvel_heroes.append('black panther')

print(marvel_heroes)

marvel_heroes.remove('black panther')

print(marvel_heroes)

marvel_heroes.insert(3, 'black panther')

print(marvel_heroes)

marvel_heroes = ['doctor strange' if hero in ['thor', 'hulk'] else hero for hero in marvel_heroes]

print(marvel_heroes)

marvel_heroes.sort()

print(marvel_heroes)