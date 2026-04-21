import sqlite3

conn = sqlite3.connect('farm.db')
print('Top 5 arugula beds by total bundles:')
for bed, total in conn.execute(
    'SELECT Bed, SUM(Bundles) FROM harvests '
    'WHERE Crop="arugula" GROUP BY Bed ORDER BY 2 DESC LIMIT 5'
).fetchall():
    print(f'  {bed}: {int(total):,} bundles')