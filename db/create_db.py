import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("db/raw_data.sqlite3")

# Create a cursor object
c = conn.cursor()

# Create table
c.execute(
    """
    CREATE TABLE model_data(
        identifierHash TEXT,
        type TEXT,
        country TEXT,
        language TEXT,
        socialNbFollowers INTEGER,
        socialNbFollows INTEGER,
        socialProductsLiked INTEGER,
        productsListed INTEGER,
        productsSold INTEGER,
        productsPassRate REAL,
        productsWished INTEGER,
        productsBought INTEGER,
        gender TEXT,
        civilityGenderId INTEGER,
        civilityTitle TEXT,
        hasAnyApp INTEGER,
        hasAndroidApp INTEGER,
        hasIosApp INTEGER,
        hasProfilePicture INTEGER,
        daysSinceLastLogin INTEGER,
        seniority INTEGER,
        seniorityAsMonths INTEGER,
        seniorityAsYears INTEGER,
        countryCode TEXT
    )
"""
)

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()
