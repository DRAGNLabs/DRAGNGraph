from os import link
from select import select
from bs4 import BeautifulSoup
import requests
import networkx as nx
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
from urllib.parse import unquote
import pickle

def save_object(obj, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported): ", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupportod):", ex)


def extract_info(urls, goal, path, graphs, tokenizer, model):
    print("Graph Loading...")
    url, soup = update_url(urls)
    if url == goal:
        return False
    curr_title = soup.find('title').contents[0]
    curr_title = curr_title[:-12]
    print(curr_title)
    paragraphs = soup.find_all("p")
    paragraphs = [paragraph for paragraph in paragraphs if paragraph.parent.has_attr('class') and paragraph.parent['class'] == ['mw-parser-output'] and not paragraph.has_attr('class')]
    if len(paragraphs) > 6:
        paragraphs = paragraphs[:6]
    valid_links, G = get_valid_links(paragraphs, curr_title, tokenizer, model)
    graphs.append(G)
    print(f"These Wikipedia pages are similar to {curr_title}:")
    for i, title in enumerate(valid_links):
        print(i, title)
    index = int(input(f"Input the index of a page that will lead to {goal[30:]}:\n"))
    selected_article = valid_links[index]
    print(f"\nYou selected {selected_article}\n")
    url = "https://en.wikipedia.org/wiki/" + selected_article
    urls.append(url)
    update_path(path, G, curr_title, title)
    return True


def update_url(urls):
    url = urls[-1]
    soup = get_soup(url)
    url = soup.find('link', rel='canonical')['href']
    urls[-1] = url
    return url, soup


def update_path(path, G, curr_title, title):
    undirected_G = G.to_undirected()
    paths = list(nx.all_simple_edge_paths(undirected_G, curr_title, title))
    if paths:
        # Find the shortest path
        shortest = paths[0]
        for path in paths:
            if len(path) < len(shortest):
                shortest = path
        path.extend(shortest)


def extract_text(paragraphs):
    text = ""
    for paragraph in paragraphs:
        text += paragraph.text
    pattern = "\[(\d+)\]"
    text = re.sub(pattern, '', text)
    return text


def get_alternate_titles(paragraphs):
    alternate_titles = []
    paragraph = paragraphs[0]
    alternate_titles = paragraph.find_all("b")
    if alternate_titles != None:
        alternate_titles = [name.string.lower() for name in alternate_titles]
    return alternate_titles


def get_valid_links(paragraphs, curr_title, tokenizer, model):
    text = extract_text(paragraphs)
    triples = get_rebel_output(text, tokenizer, model)
    for triple in triples:
        triple['head'] = triple['head'].lower()
        triple['tail'] = triple['tail'].lower()
    links_titles = get_links_titles(paragraphs)
    # Modify triples to reflect the current article's title and next articles' titles
    triples = update_triples(triples, paragraphs, curr_title, links_titles)
    G = make_graph(triples)
    undirected_G = G.to_undirected()
    # Search the graph for links that can trace back to the root
    #show_graph(undirected_G)
    valid_links = []
    if undirected_G.has_node(curr_title):
        for _, title in links_titles.items():
            paths = list(nx.all_simple_paths(undirected_G, curr_title, title))
            if paths:
                valid_links.append(title)
    return valid_links, G


def update_triples(triples, paragraphs, curr_title, links_titles):
    alternate_titles = get_alternate_titles(paragraphs)
    for triple in triples:
        if triple['head'] in alternate_titles or curr_title.lower() == triple['head']:
            triple['head'] = curr_title
        if triple['tail'] in alternate_titles or curr_title.lower() == triple['tail']:
            triple['tail'] = curr_title
        if triple['head'] in links_titles.keys():
            triple['head'] = links_titles[triple['head']]
        if triple['tail'] in links_titles.keys():
            triple['tail'] = links_titles[triple['tail']]
    return triples


def get_links_titles(paragraphs):
    links_titles = {}
    for paragraph in paragraphs:
        sub_links = paragraph.find_all("a")
        for sub_link in sub_links:
            if sub_link.has_attr('href') and "/wiki/" in sub_link['href'] and sub_link.has_attr('title'):
                links_titles[sub_link['title'].lower()] = unquote(sub_link['href'][6:].replace("_", " "))
    return links_titles


def make_graph(triples):
    G = nx.DiGraph()
    triples_graph = [(triple['head'], triple['tail'], {'relation': triple['type']}) for triple in triples]
    G.add_edges_from(triples_graph)
    return G


def get_rebel_output(text, tokenizer, model):
    gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
    }
    triples = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        # Tokenizer text
        model_inputs = tokenizer(paragraph, max_length=1024, padding=True, truncation=True, return_tensors = 'pt')

        # Generate
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            **gen_kwargs,
        )

        # Extract text
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # Extract triplets
        for i, sentence in enumerate(decoded_preds):
            triples.extend(extract_triplets(sentence))
    return triples


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


def get_soup(url):
    headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    page = requests.get(url, headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def show_graph(G):
    edge_labels = nx.get_edge_attributes(G, "relation")
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, pos=pos, connectionstyle='arc3, rad = 0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()  # pause before exiting


def main():
    goal = "https://en.wikipedia.org/wiki/Termessadou-Dibo"
    urls = ['https://en.wikipedia.org/wiki/Guinea']
    path = []
    graphs = []
    print("""\nWelcome to Wiki Chase 2.0!  
    You will be asked to select a page that relates to the current page. 
    Try to create a path that ends at the goal page. Good luck!\n""")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    # Play the game
    while extract_info(urls, goal, path, graphs, tokenizer, model):
        pass
    print("You Won!")
    for url in urls:
        print(url)
    for connection in path:
        print(connection)

    # Combine all graphs together
    I = nx.DiGraph()
    for graph in graphs:
        I = nx.compose(I, graph)

    # Create a subgraph of nodes connected to the goal's title
    goal_title = goal[30:].replace("_", " ")
    I = I.subgraph(nx.shortest_path(I.to_undirected(), goal_title))
    show_graph(I)

def extract_and_save(urls, tokenizer, model):
    print("Graph Loading...")
    url, soup = update_url(urls)
    curr_title = soup.find('title').contents[0]
    curr_title = curr_title[:-12]
    print(curr_title)
    paragraphs = soup.find_all("p")
    paragraphs = [paragraph for paragraph in paragraphs if paragraph.parent.has_attr('class') and paragraph.parent['class'] == ['mw-parser-output'] and not paragraph.has_attr('class')]
    if len(paragraphs) > 6:
        paragraphs = paragraphs[:6]
    valid_links, G = get_valid_links(paragraphs, curr_title, tokenizer, model)
    save_object(G, './{}.pkl'.format(curr_title))

def save_wikipage():
    goal = ""
    urls = ['https://en.wikipedia.org/wiki/Guinea']
    path = []
    graphs = []
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    extract_and_save(urls, tokenizer, model)

if __name__ == "__main__":
    #main()
    save_wikipage()