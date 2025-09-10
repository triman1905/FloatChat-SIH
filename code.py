"""1. ARGO NetCDF Data Ingestion	Parsed ARGO NetCDF files to extract oceanographic variables (latitude, longitude, depth, temperature, salinity, date).	Completed
2. Data Transformation & Loading	Transformed extracted data into tabular format and loaded it into a Neon PostgreSQL database.	Completed
3. Metadata Summary Generation	Created concise textual summaries describing each ARGO profile’s key metadata (location, date, temperature and salinity ranges)."""



import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Neon/PostgreSQL connection string
DATABASE_URL = "postgresql://neondb_owner:npg_koHC7hDGFx2p@ep-dark-hall-adj44gpn-pooler.c-2.us-east-1.aws.neon.tech/argo_data?sslmode=require&channel_binding=require"
engine = create_engine(DATABASE_URL)

def create_argo_table(engine):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS argo_float_data (
        id SERIAL PRIMARY KEY,
        latitude FLOAT,
        longitude FLOAT,
        depth FLOAT,
        temperature FLOAT,
        salinity FLOAT,
        date DATE
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()

def parse_argo_netcdf(file_path):
    nc_file = netCDF4.Dataset(file_path)
    latitudes = nc_file.variables['LATITUDE'][:]
    longitudes = nc_file.variables['LONGITUDE'][:]
    depths = nc_file.variables['PRES'][:]
    temperatures = nc_file.variables['TEMP'][:]
    salinities = nc_file.variables['PSAL'][:]
    juld = nc_file.variables['JULD'][:]
    
    ref_date = datetime(1950, 1, 1)
    dates = [ref_date + timedelta(days=float(d)) for d in juld]
    
    nc_file.close()
    return latitudes, longitudes, depths, temperatures, salinities, dates

def flatten_data_for_db(latitudes, longitudes, depths, temperatures, salinities, dates):
    data_rows = []
    num_profiles = latitudes.shape[0]
    num_levels = depths.shape[1]
    
    for i in range(num_profiles):
        for j in range(num_levels):
            temp = temperatures[i][j]
            salt = salinities[i][j]
            depth = depths[i][j]
            
            if pd.isna(temp) or pd.isna(salt) or pd.isna(depth):
                continue
            
            data_rows.append({
                'latitude': float(latitudes[i]),
                'longitude': float(longitudes[i]),
                'depth': float(depth),
                'temperature': float(temp),
                'salinity': float(salt),
                'date': dates[i].date()
            })
    
    df = pd.DataFrame(data_rows)
    return df

def generate_metadata_summaries(latitudes, longitudes, temperatures, salinities, dates):
    summaries = []
    num_profiles = latitudes.shape[0]
    num_levels = temperatures.shape[1]

    for i in range(num_profiles):
        temps = temperatures[i]
        salts = salinities[i]
        
        valid_temps = temps[~pd.isna(temps)]
        valid_salts = salts[~pd.isna(salts)]
        
        if len(valid_temps) == 0 or len(valid_salts) == 0:
            continue
        
        temp_min, temp_max = valid_temps.min(), valid_temps.max()
        salt_min, salt_max = valid_salts.min(), valid_salts.max()
        
        summary = (
            f"Profile near {latitudes[i]:.2f}°, {longitudes[i]:.2f}° on {dates[i].date()} "
            f"with temperature range {temp_min:.2f}–{temp_max:.2f} °C "
            f"and salinity range {salt_min:.2f}–{salt_max:.2f} PSU."
        )
        summaries.append(summary)
    return summaries

def ingest_argo_netcdf_to_neon(nc_file_path):
    print("Creating ARGO data table if it doesn't exist...")
    create_argo_table(engine)

    print(f"Parsing NetCDF file: {nc_file_path}")
    latitudes, longitudes, depths, temperatures, salinities, dates = parse_argo_netcdf(nc_file_path)

    print("Transforming data for database insertion...")
    df = flatten_data_for_db(latitudes, longitudes, depths, temperatures, salinities, dates)
    print(f"Rows to insert: {len(df)}")

    print("Inserting data into Neon database...")
    df.to_sql('argo_float_data', engine, if_exists='append', index=False, method='multi')
    print("Data insertion complete.")

    print("Generating metadata summaries for profiles...")
    summaries = generate_metadata_summaries(latitudes, longitudes, temperatures, salinities, dates)
    print(f"Generated {len(summaries)} metadata summaries.")
    # Save summaries if desired (e.g., to file) or integrate into vector DB next
    for s in summaries[:5]:
        print("-", s)

if __name__ == "__main__":
    # Replace with your local NetCDF file path
    INF_NETCDF_FILE = "/content/D20250610_prof_0.nc"
    ingest_argo_netcdf_to_neon(INF_NETCDF_FILE)


"""4. Embedding & FAISS Vector Index	
Generated sentence embeddings for metadata summaries and built a FAISS vector index to enable semantic search and retrieval."""

# Install required packages before running:
# pip install faiss-cpu sentence-transformers numpy --quiet

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Example metadata summaries (replace or load your actual summaries)


metadata_summaries = [
    "Profile near -12.98°, 0.92° on 2010-04-06 with temperature range 8.74–26.52 °C and salinity range 34.78–36.67 PSU.",
    "Profile near -13.03°, -0.98° on 2010-04-15 with temperature range 3.97–26.14 °C and salinity range 34.49–36.69 PSU.",
    "Profile near -13.04°, 1.05° on 2010-04-25 with temperature range 4.00–25.46 °C and salinity range 34.49–36.72 PSU.",
    "Profile near -12.97°, 1.21° on 2010-05-05 with temperature range 3.96–24.81 °C and salinity range 34.49–36.82 PSU.",
    "Profile near -12.95°, 1.39° on 2010-05-15 with temperature range 3.96–25.19 °C and salinity range 34.49–36.79 PSU."
]



def build_faiss_index(summaries):
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = model.encode(summaries, convert_to_numpy=True)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index

def search_faiss(query, model, index, summaries, top_k=5):
    # Embed the query
    query_vec = model.encode([query], convert_to_numpy=True)
    # Search
    distances, indices = index.search(query_vec, top_k)
    # Fetch results
    results = [summaries[i] for i in indices[0] if i < len(summaries)]
    return results

if __name__ == "__main__":
    print("Building FAISS index from metadata summaries...")
    model, faiss_index = build_faiss_index(metadata_summaries)
    print(f"Index contains {faiss_index.ntotal} entries.")

    # Example search query
    user_query = "Show profiles with high salinity near the equator"
    print(f"\nSearching for: '{user_query}'")
    matches = search_faiss(user_query, model, faiss_index, metadata_summaries, top_k=3)

    print("\nTop matches:")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match}")



# faiss data retrieval

from sentence_transformers import SentenceTransformer
import faiss
from sqlalchemy import create_engine, text
import numpy as np

# =======================
# Configuration Variables
# =======================
DATABASE_URL = "postgresql://neondb_owner:npg_koHC7hDGFx2p@ep-dark-hall-adj44gpn-pooler.c-2.us-east-1.aws.neon.tech/argo_data?sslmode=require&channel_binding=require"
TABLE_NAME = "argo_float_data"

# =====================
# Metadata Summaries (Example)
# =====================
metadata_summaries = [
    "Profile near -12.98°, 0.92° on 2010-04-06 with temperature range 8.74–26.52 °C and salinity range 34.78–36.67 PSU.",
    "Profile near -13.03°, -0.98° on 2010-04-15 with temperature range 3.97–26.14 °C and salinity range 34.49–36.69 PSU.",
    "Profile near -13.04°, 1.05° on 2010-04-25 with temperature range 4.00–25.46 °C and salinity range 34.49–36.72 PSU.",
    "Profile near -12.97°, 1.21° on 2010-05-05 with temperature range 3.96–24.81 °C and salinity range 34.49–36.82 PSU.",
    "Profile near -12.95°, 1.39° on 2010-05-15 with temperature range 3.96–25.19 °C and salinity range 34.49–36.79 PSU."
]

# ==============
# FAISS Functions
# ==============
def build_faiss_index(summaries):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(summaries, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

def search_faiss(query, model, index, summaries, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = [summaries[i] for i in indices[0] if i < len(summaries)]
    return results

# ==========================
# Database Query Execution
# ==========================
engine = create_engine(DATABASE_URL)

def run_sql_with_cache(sql):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        return [f"SQL execution error: {e}"]

# ================================
# Improved Rule-Based SQL Generator
# ================================
class SimpleSQLGenerator:
    def generate_content(self, model, contents):
        user_question = ""
        try:
            user_question = contents.split("User question:\n")[-1].split("\n")[0].lower()
        except:
            pass
        
        if "temperature above 20" in user_question:
            sql = f"SELECT latitude, longitude, depth, temperature, salinity, date FROM {TABLE_NAME} WHERE temperature > 20 LIMIT 10;"
        elif "within 5 degrees latitude" in user_question or "latitude" in user_question:
            sql = f"SELECT latitude, longitude, depth, temperature, salinity, date FROM {TABLE_NAME} WHERE latitude BETWEEN 5 AND 15 LIMIT 10;"
        elif "salinity" in user_question:
            sql = f"SELECT latitude, longitude, depth, temperature, salinity, date FROM {TABLE_NAME} WHERE salinity > 35 LIMIT 10;"
        else:
            sql = f"SELECT latitude, longitude, depth, temperature, salinity, date FROM {TABLE_NAME} LIMIT 5;"
        
        class Response:
            text = sql
        return Response()

simple_sql_generator = SimpleSQLGenerator()

# ===========================
# Pipeline with Simple SQL Generator
# ===========================
def generate_sql_with_context(user_query, retrieved_context, sql_generator):
    prompt = (
        "You are an expert in ARGO oceanographic data.\n"
        "Use the following metadata summaries as context to generate an accurate SQL query.\n\n"
        f"Metadata summaries:\n{retrieved_context}\n\n"
        f"User question:\n{user_query}\n\n"
        "Please provide ONLY the SQL query."
    )
    response = sql_generator.generate_content(
        model="dummy-model",
        contents=prompt
    )
    return response.text.strip()

def conversational_query_pipeline(user_query, model, faiss_index, summaries, sql_generator):
    greetings = ["hello", "hi", "hey"]
    if user_query.lower().strip() in greetings:
        return "Chatbot", ["Hello! You can ask me about ARGO oceanographic data."]
    
    retrieved_context = "\n".join(search_faiss(user_query, model, faiss_index, summaries, top_k=3))
    sql = generate_sql_with_context(user_query, retrieved_context, sql_generator)
    results = run_sql_with_cache(sql)
    if not results:
        return "Chatbot", ["Sorry, no data found or query could not be executed."]
    return "Chatbot", results

# ======================
# Chatbot Interactive Loop
# ======================
def chatbot_loop():
    print("Welcome to ARGO Retrieval-Augmented Chatbot! Type 'exit' to quit.")
    model, faiss_index = build_faiss_index(metadata_summaries)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            speaker, response = conversational_query_pipeline(user_input, model, faiss_index, metadata_summaries, simple_sql_generator)
            print(f"{speaker}:")
            for row in response:
                print(row)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot_loop()
