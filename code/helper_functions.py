import os
from openai import OpenAI
from dotenv import load_dotenv
import csv
import pandas as pd  # Import pandas 
from IPython.display import display, HTML  # Import HTML
import ipywidgets as widgets  # Import ipywidgets
import random
import gradio as gr
import requests
import json
import folium
import panel as pn


# Load environment variables from a .env file
load_dotenv('.env', override=True)

# Retrieve the base path from environment variables
base_path = os.getenv('BASE_PATH')

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is available
if not openai_api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Function to get a completion using the updated SDK
def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# Function to get a completion from messages using the updated SDK
def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# Define the input widget
inp = pn.widgets.TextInput(name='Enter your message:', placeholder='Type here...')

def collect_messages(event, context, panels):
    prompt = inp.value
    inp.value = ''
    context.append({'role': 'user', 'content': f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role': 'assistant', 'content': f"{response}"})

    # Append User's input to panels
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600))
    )
    # Append Assistant's response with custom background styling
    panels.append(
        pn.Row(
            'Assistant:',
            pn.pane.Markdown(f"<div style='background-color: #F6F6F6; padding: 10px;'>{response}</div>", width=600)
        )
    )

    # Return the updated chat display
    return pn.Column(*panels, height=500, scroll=True)



def read_csv(file_path):
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The contents of the CSV file as a DataFrame.
    """
    return pd.read_csv(file_path)

def get_dog_age(human_age):
    """This function takes one parameter: a person's age as an integer and returns their age if
    they were a dog, which is their age divided by 7. """
    return human_age / 7

def get_goldfish_age(human_age):
    """This function takes one parameter: a person's age as an integer and returns their age if
    they were a dog, which is their age divided by 5. """
    return human_age / 5

def get_cat_age(human_age):
    if human_age <= 14:
        # For the first 14 human years, we consider the age as if it's within the first two cat years.
        cat_age = human_age / 7
    else:
        # For human ages beyond 14 years:
        cat_age = 2 + (human_age - 14) / 4
    return cat_age

def get_random_ingredient():
    """
    Returns a random ingredient from a list of 20 smoothie ingredients.
    
    The ingredients are a bit wacky but not gross, making for an interesting smoothie combination.
    
    Returns:
        str: A randomly selected smoothie ingredient.
    """
    ingredients = [
        "rainbow kale", "glitter berries", "unicorn tears", "coconut", "starlight honey",
        "lunar lemon", "blueberries", "mermaid mint", "dragon fruit", "pixie dust",
        "butterfly pea flower", "phoenix feather", "chocolate protein powder", "grapes", "hot peppers",
        "fairy floss", "avocado", "wizard's beard", "pineapple", "rosemary"
    ]
    
    return random.choice(ingredients)

def get_random_number(x, y):
    """
        Returns a random integer between x and y, inclusive.
        
        Args:
            x (int): The lower bound (inclusive) of the random number range.
            y (int): The upper bound (inclusive) of the random number range.
        
        Returns:
            int: A randomly generated integer between x and y, inclusive.

        """
    return random.randint(x, y)

def calculate_llm_cost(characters, input_price_per_1M_tokens=2.50, output_price_per_1M_tokens=10.00):
    """
    Calculate the cost of using the LLM based on the number of characters and the price per 1M tokens.

    Args:
        characters (int): The number of characters in the input.
        input_price_per_1M_tokens (float): The price per 1M input tokens. Default is 2.50.
        output_price_per_1M_tokens (float): The price per 1M output tokens. Default is 10.00.

    Returns:
        str: The cost formatted as a string with 4 decimal places.
    """
    tokens = characters / 4  # Assuming 1 token is approximately 4 characters
    input_cost = (tokens / 1_000_000) * input_price_per_1M_tokens
    output_cost = (tokens / 1_000_000) * output_price_per_1M_tokens
    total_cost = input_cost + output_cost
    return f"${total_cost:.4f}"

def print_llm_response(prompt):
    """
    This function takes a prompt as input, which must be a string enclosed in quotation marks,
    and passes it to OpenAI’s GPT-4o-mini model. The function then prints the response of the model.
    
    Args:
        prompt (str): The input prompt to be sent to the GPT-4o-mini model.
    
    Raises:
        ValueError: If the input prompt is not a string.
        TypeError: If there is an error in processing the response.
    """
    try:
        if not isinstance(prompt, str):
            raise ValueError("Input must be a string enclosed in quotes.")
        
        # Create a completion request to the GPT-4 model
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful but terse AI assistant who gets straight to the point.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        
        # Extract and print the response from the model
        response = completion.choices[0].message.content
        print("*" * 100)
        print(response)
        print("*" * 100)
        print("\n")
    except TypeError as e:
        print("Error:", str(e))

def get_llm_response(prompt):
    """
    This function takes a prompt as input, which must be a string enclosed in quotation marks,
    and passes it to OpenAI’s GPT-4o-mini model. The function then returns the response of the model as a string.
    
    Args:
        prompt (str): The input prompt to be sent to the GPT-4o-mini model.
    
    Returns:
        str: The response from the GPT-4o-mini model.
    """
    # Create a completion request to the GPT-4o-mini model
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful but terse AI assistant who gets straight to the point.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    
    # Extract and return the response from the model
    response = completion.choices[0].message.content
    return response


def open_chatbot():
    """
    This function opens a Gradio chatbot window that is connected to OpenAI's GPT-4o-mini model.
    """
    gr.close_all()
    gr.ChatInterface(fn=get_chat_completion).launch(quiet=True)

def read_journal(file_path):
    """
    Reads the content of a journal file and returns it as a string.
    
    Args:
        file_path (str): The path to the journal file.
    
    Returns:
        str: The content of the journal file.
    
    Raises:
        FileNotFoundError: If the file is not found.
        Exception: If any other error occurs while reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define a function to format the response into HTML
def format_response_to_html(response):
    """
    This function formats the given response into an HTML structure.
    
    Args:
        response (str): The response text to be formatted into HTML.
    
    Returns:
        str: The formatted HTML content.
    """
    # Create the HTML content with the response embedded
    html_content = f"""
    <html>
    <body>
    <p>Embarking on a gastronomic journey through Cape Town revealed a city brimming with culinary treasures. Each stop was a testament to the rich flavors and unique dishes that define this vibrant city's food scene.</p>
    {response}
    </body>
    </html>
    """
    return html_content

def upload_txt_file(file_path):
    """
    Uploads a text file and returns its content as a string.
    
    Args:
        file_path (str): The path to the text file.
    
    Returns:
        str: The content of the text file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def upload_txt_file_widget():
    """
    Uploads a text file and saves it to the specified directory.
    
    Args:
        directory (str): The directory where the uploaded file will be saved. 
        Defaults to the current working directory.
    """
    # Create the file upload widget
    upload_widget = widgets.FileUpload(
        accept='.txt',  # Accept text files only
        multiple=False  # Do not allow multiple uploads
    )
    # Impose file size limit
    output = widgets.Output()
    
    # Function to handle file upload
    def handle_upload(change):
        with output:
            output.clear_output()
            # Read the file content
            content = upload_widget.value[0]['content']
            name = upload_widget.value[0]['name']
            size_in_kb = len(content) / 1024
            
            if size_in_kb > 3:
                print(f"Your file is too large, please upload a file that doesn't exceed 3KB.")
                return
		    
            # Save the file to the specified directory
            with open(name, 'wb') as f:
                f.write(content)
            # Confirm the file has been saved
            print(f"The {name} file has been uploaded.")

    # Attach the file upload event to the handler function
    upload_widget.observe(handle_upload, names='value')

    display(upload_widget, output)

def list_files_in_directory(directory_path):
    """
    Lists all files in a given directory.
    
    Args:
        directory_path (str): The path to the directory.
    
    Returns:
        list: A list of file names in the directory.
    """
    try:
        files = os.listdir(directory_path)
        return files
    except FileNotFoundError:
        print(f"Error: The directory {directory_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_file(file_name):
    """
    Generates an HTML button to download the specified file.
    
    Args:
        file_name (str): The name of the file to be downloaded.
    
    Returns:
        None
    """
    html = f"""
    <html>
    <body>
    <a href="{file_name}" download="{file_name}">
        <button>Click here to download your file</button>
    </a>
    </body>
    </html>
    """
    display(HTML(html))

def display_table(data):
    """
    Displays a list of dictionaries as an HTML table.
    
    Args:
        data (list): A list of dictionaries representing the table data.
    
    Returns:
        None
    """
    if not data:
        print("No data to display.")
        return
    
    # Extract headers from the first dictionary
    headers = data[0].keys()
    
    # Create the HTML table
    html = "<table border='1'>"
    html += "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
    
    for row in data:
        html += "<tr>" + "".join(f"<td>{row[header]}</td>" for header in headers) + "</tr>"
    
    html += "</table>"
    
    # Display the HTML table
    display(HTML(html))

def read_csv_dict(csv_file_path):
    """This function takes a csv file and loads it as a dict."""

    # Initialize an empty list to store the data
    data_list = []

    # Open the CSV file
    with open(csv_file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
    
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append the row to the data list
            data_list.append(row)

    # Convert the list to a dictionary
    data_dict = {i: data_list[i] for i in range(len(data_list))}
    return data_dict

def display_table_pd(data):
    df = pd.DataFrame(data)

    # Display the DataFrame as an HTML table
    display(HTML(df.to_html(index=False)))

def get_current_time():
    now = dt.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    print(f"{fahrenheit}°F is equivalent to {celsius:.2f}°C")
    
def celsius_to_fahrenheit(celsius):
    fahrenheit = celsius * 9 / 5 + 32 
    print(f"{celsius}°C is equivalent to {fahrenheit:.2f}°F")

def beautiful_barh(labels, values):
	# Create the bar chart
	plt.figure(figsize=(9, 5))
	bars = plt.barh(labels, values, color = plt.cm.tab20.colors)

	for bar in bars:
		plt.text(bar.get_width()/2,   # X coordinate 
			 bar.get_y() + bar.get_height()/2,  # Y coordinate 
			 f'${bar.get_width() / 1e9:.1f}B',  # Text label 
			 ha='center', va='center', color='w', fontsize=10, fontweight = "bold")
			 
	# Customizing the x-axis to display values in billions
	def billions(x, pos):
		"""The two args are the value and tick position"""
		return f'${x * 1e-9:.1f}B'

	formatter = FuncFormatter(billions)
	plt.gca().xaxis.set_major_formatter(formatter)


	# Inverting the y-axis to have the highest value on top
	plt.gca().invert_yaxis()

def display_map():
    # Define the bounding box for the continental US
    us_bounds = [[24.396308, -125.0], [49.384358, -66.93457]]
    # Create the map centered on the US with limited zoom levels
    m = folium.Map(
	    location=[37.0902, -95.7129],  # Center the map on the geographic center of the US
	    zoom_start=5,  # Starting zoom level
	    min_zoom=4,  # Minimum zoom level
	    max_zoom=10,
	    max_bounds=True,
	    control_scale=True  # Maximum zoom level
	)

    # Set the bounds to limit the map to the continental US
    m.fit_bounds(us_bounds)
    # Add a click event to capture the coordinates
    m.add_child(folium.LatLngPopup())
    title_html = '''
	<div style="
	display: flex;
	justify-content: center;
	align-items: center;
	width: 100%; 
	height: 50px; 
	border:0px solid grey; 
	z-index:9999; 
	font-size:30px;
	padding: 5px;
	background-color:white;
	text-align: center;
	">
	&nbsp;<b>Click to view coordinates</b>
	</div>
	'''
	
    m.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    return m
    
def get_forecast(lat, lon):
    url = f"https://api.weather.gov/points/{lat},{lon}"

    # Make the request to get the grid points
    response = requests.get(url)
    data = response.json()
    # Extract the forecast URL from the response
    forecast_url = data['properties']['forecast']

    # Make a request to the forecast URL for the selected location
    forecast_response = requests.get(forecast_url)
    forecast_data = forecast_response.json()
    
    daily_forecast = forecast_data['properties']['periods']
    return daily_forecast

def print_journal(file):
    with open(file, "r") as f:
        journal = f.read()
    print(journal)

def create_bullet_points(file):
    # Read in the file and store the contents as a string
    with open(file, "r") as f:
        file_contents = f.read()

    # Write a prompt and pass to an LLM
    prompt = f"""Please summarize the following text into three bullet points:
    
    {file_contents}
    """
    bullets = get_llm_response(prompt)  # Pass the prompt to the LLM

    # Return the bullet points
    return bullets
