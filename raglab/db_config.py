DB_CONFIG = {
    "database": "raglab",
    "host": "localhost",
    "port": 5432,
    "user": "raguser",
    "password": "ragpass",
    "embed_dim": 384,
}

DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
