# myUniHackathon
The Streamlit application, utilizes public information from University of Toronto to support students when searching for information about the different programs at the university. 
If you want to include public information from other Universities, simply supply them as a text file into the u_data subfolder. 

It utilizes Langchain, a Chroma vector database, and Cohere LLM. 
A Cohere API Key is required.

To run this program locally, add a .streamlit subfolder, and then a secrets.toml file in that subfolder.

The file should contain a single line: COHERE_API_KEY="<i><b>ENTER YOUR COHERE API KEY HERE IN BETWEEN THE DOUBLE QUOTES</b></i>"

To run this on Streamlit Cloud, simply add a SECRET of the same: COHERE_API_KEY="<i><b>ENTER YOUR COHERE API KEY HERE IN BETWEEN THE DOUBLE QUOTES</b></i>"

If you want to update any of the Langchain and/or Cohere Default parameters, or the UI verbiage, those can be modified in the config.ini file

