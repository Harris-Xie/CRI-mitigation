import fasttext
import numpy as np
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
import cufflinks as cf
from transformers import AutoTokenizer
import warnings
from bs4 import BeautifulSoup
import codeprep.api.text as cp
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
import io

from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
from azure.storage.blob import BlobClient
from azure.kusto.ingest import (
    BlobDescriptor,
    FileDescriptor,
    IngestionProperties,
    IngestionStatus,
    KustoStreamingIngestClient,
    ManagedStreamingIngestClient,
    QueuedIngestClient,
    StreamDescriptor,
)
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text
def text_clean(text):# used for fasttext
    cache_english_stopwords = stopwords.words('english')
    cache_english_stopwords+=['!', ',', '.', '?', '-s', '-ly', '</s> ', 's','[',']',':','(',')','{','}','\'','<','>','+','-','_','__','--','|','\'\'']
    # Remove HTML tags (e.g. &amp;)
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    # Remove certain value symbols
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    # Remove hyperlinks
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    # # Remove some specialized abbreviation words, in other words, words with fewer letters
    string4 = " ".join(re.findall(r'\b\w+\b', text_no_hyperlinks))
    # Tokenization
    tokens = word_tokenize(string4)
    # Remove stopwords
    list_no_stopwords = [i for i in tokens if i not in cache_english_stopwords]
    # Final filtered result
    text_filtered = ' '.join(list_no_stopwords)  # ''.join() would join without spaces between words.
    return text_filtered
def count_word_num(text):
    if text is np.nan:
        return 0
    else:
        return len(text.split())
def remove_numbers(text):
    cleaned_text = re.sub(r'\d+', '', text)
    return cleaned_text
def truncate_text(text, length):
    if text is np.nan:
        return text
    else:
        return ' '.join(text.split()[:length])
def combine(x,column_list):
    res=''
    for column in column_list:
        res+=str(x[column])
    return res
def remove_tabs_newlines(text):
    text_without_newlines_and_tabs = text.replace('\n', '').replace('\t', '')
    return text_without_newlines_and_tabs
def trainingFile_preprocess(df,column_list,target,test=False):
    label_list=df[target].apply(str)
    document_list=df['metadata']
    text_filtered_list=document_list.apply(lambda x: combine(x,column_list))
    text_filtered_list=text_filtered_list.apply(remove_tabs_newlines)
    if not test:
        file=np.array('__label__'+label_list+' '+text_filtered_list)
    else:
        file=np.array(text_filtered_list)
    return file

def clean(text,max_len=300):
    content = text
    content=remove_html_tags(content)
    content=remove_numbers(content)
    # Replace all whitespaces
    content = re.sub(r'\&\w*;|#\w*|@\w*', '', content)
    content = content.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\\\', ' ')
    content = content.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\\', ' ')
    # Remove all quotes and brackets
    content = content.replace('\"', ' ').replace("\'", ' ').replace('(', ' ').replace(')', ' ')
    # Remove all vertical bars
    content = content.replace('|', ' ')
    # Replace all consecutive '-'s with only one '-'
    content = re.sub('-+', '-', content)
    # Remove filenames if its extension is in 'file_exts.txt'
    common_file_extensions = open('Data_Models/file_exts.txt', 'r').read().splitlines()
    content = ' '.join([word for word in content.split() if '.' + word.split('.')[-1] not in common_file_extensions])
    # If there are multiple whitespaces, replace with one whitespace
    content =  re.sub(' +', ' ', content)
    content = re.sub(r"http\S+", '', content)
    content = re.sub(r'\w{8}-\w{4}-\w{4}-\w{4}-\w{12}', ' ', content)
    content = truncate_text(content, max_len) if count_word_num(content) > max_len else content
    content = content.replace(' ','')
    # content = content[:6000]
    return content

def data_clean(df):
    new_columns = {}
    for column in df.columns:
        if column.startswith("max_ModifiedDate_"):
            new_column = column.replace("max_ModifiedDate_", "")
            new_columns[column] = new_column
    df.rename(columns=new_columns, inplace=True)
    null_ratio = df.isnull().mean()
    columns_to_drop = null_ratio[null_ratio > 0.1].index
    df = df.drop(columns=columns_to_drop)
    # df['Summary']=df['Summary'].apply(html2plaintext)
    df['Text'] = df['Text'].apply(html2plaintext)
    df['Title'] = df['Title'].apply(parse_id)
    df['Title'] = df['Title'].apply(html2plaintext)
    # df['Summary']=df['Summary'].apply(remove_tabs_newlines)
    df = df[['IncidentId','CreateDate','OwningTeamId', 'RoutingId', 'Severity', 'Title', 'IsPurged', 'Text']]
    df_sorted = df.sort_values('CreateDate')
    return df_sorted
def process_text(html):
    soup = BeautifulSoup(html, 'lxml')

    h2_node = soup.find('h2', string=' Node Story ')
    if not h2_node:
        h2_node = soup.find('h2', string='Node Story')

    # Find <table> node after the <h2> node
    table = h2_node.find_next('table')

    # Find all <tr> nodes
    rows = table.find_all('tr')

    # Extract the header (first <tr> node)
    header_row = rows[0]
    headers = [header.text for header in header_row.find_all('td')]

    # Extract data rows
    data_rows = rows[1:]
    data = []

    for row in data_rows:
        cells = {}
        for i, cell in enumerate(row.find_all('td')):
            cells[headers[i]] = (re.sub(r'http\S+|https\S+', '', cell.text))
        data.append(cells)
    return str(data)
def html2plaintext(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def parse_id(text):
    for datetime_pattern in datetime_patterns:
        text = datetime_pattern.sub('TIMESTAMP', text)
    text = ipv6_pattern.sub('IP', text)
    words = text.split(' ')
    res = []
    for word in words:
        if count_underscore(word) > 1:
            subwords = re.split('_', word)
        else:
            subwords = [word]
        for subword in subwords:
            if len(subword) <= 3:
                res.append(subword)
                continue
            subword = version_pattern.sub(lambda x: str(x.group(1)) + ' VERSION', subword)
            subword = id_pattern_ethernet.sub(' EID', subword)
            subword = id_pattern_long.sub('LONGID', subword)
            subword = id_pattern_2.sub('IID', subword)
            subword = id_pattern_short.sub('SID', subword)
            subword = sr_pattern.sub('SR', subword)
            subword = ip_pattern.sub('IP', subword)
            subword = id_pattern.sub('ID', subword)
            if subword in ['ID', 'IED', 'IID', 'SR']:
                res.append(subword)
                continue
            if "'" in subword and len(subword) < 10:
                res.append(subword)
                continue
            res.append(' '.join(cp.basic(subword)))
    return (' '.join(res)).lower()

def count_underscore(word):
    return sum([1 for char in word if char == '_'])

datetime_patterns = [
    re.compile(r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\.\d{3})?'),
    re.compile( r'\d{,2}/\d{,2}/\d{4} \d{,2}:\d{2}:\d{2} [AP]M( [+-]\d\d:\d\d \([a-zA-Z]{2,3}\))?' ), # 3/24/2022 5:30:00 PM
]

id_pattern = re.compile( r'(?=.*\d.*)(?=.*[a-zA-Z].*)[\w\d]+([\._-][\w\d]+){2,}' )
id_pattern_2 = re.compile( r'[a-zA-Z]{2,3}\d{1,2}(edge|AzSet|Prd[a-zA-Z]{3})\d{1,2}' )
id_pattern_ethernet = re.compile( r'([Ee]thernet|([Hh]undred|forty|[Tt]en)gig[eE])\d{,3}/\d{,3}(/\d{,3}){,2}' )
id_pattern_long = re.compile( r'(?=.*\d.*)(?=.*[a-zA-Z].*)\w{25,}' )
id_pattern_short = re.compile( r'[a-zA-Z]{3}\d{2}\.[a-zA-Z]{3}\d{2}' )
ip_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(/\d{,2})?')
ipv6_pattern = re.compile(r'(?=.*\w.*)\w{,4}?:\w{,4}?:\w{,4}?:\w{,4}?:\w{,4}?:\w{,4}?')
version_pattern = re.compile( r'([a-zA-Z]+)\d+\.\d+\.\d+' )
sr_pattern = re.compile( r'\d{15,16}' )

re_delimiters = '|'.join([re.escape(x) for x in '/ - _'.split()])

KUSTO_INGESTDB = "gandalf_deepADDev"
KUSTO_INGEST_TABLE = "RCA_Copilot"
def setup_kusto_client(cluster, ingest = False):
    # get client id and secret from key vault
    appid = ""
    appkey = ""
    authority_id = ""
    # generate client
    if ingest:
        cluster = "ingest-"+cluster
    cluster = f"https://{cluster}.kusto.windows.net/"
    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, appid, appkey, authority_id)
    # kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, client_id, client_secret, authority_id)
    if ingest:
        return QueuedIngestClient(kcsb)
    return KustoClient(kcsb)
def ingest_kusto(client, data):
    ingestion_props = IngestionProperties(
        database=KUSTO_INGESTDB,
        table=KUSTO_INGEST_TABLE,
    )
    client.ingest_from_dataframe(data, ingestion_properties=ingestion_props)
def inject(df):
    cluster = "https://ingest-gandalfdeepad.kusto.windows.net"

    # In case you want to authenticate with AAD application.
    client_id = ""
    client_secret = ""

    # read more at https://docs.microsoft.com/en-us/onedrive/find-your-office-365-tenant-id
    authority_id = ""

    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, client_id, client_secret,
                                                                             authority_id)
    client = KustoStreamingIngestClient(kcsb)
    ingestion_properties = IngestionProperties(database="gandalf_deepADDev",table="RCA_Copilot",
                                               data_format=DataFormat.CSV)
    client.ingest_from_dataframe(df, ingestion_properties=ingestion_properties)
    print('Done data to kusto')
def IdToTeam(Id):
    table=r"""70906:RDOS\AzureHostOSDRINinjas-RainierCodeYellow
32039:WINDOWSCLOUDSERVERDEVELOPMENT\Triage
77987:HOSTNETWORKING\HostNICandRDMATriage
78672:XSTORE\XDiskAvailability
35720:RDOS\Azure-Host-Os-Sev-3-4
77632:NETWORKDATAPATH\Triage
72125:ONEFLEETNODE\APlat_ProdIntellgenceLowSev
77618:AZUREHOSTSTORAGE\StorageClient
34137:AZURECOMPUTEINSIGHTS\Gandalf
49976:ONEFLEETNODE\AzureHost-Agent-Sev-3-4
51270:RDOS\OSBaseVirtualization
24531:RDOS\AzureHost-Os-Sev-1-2
69328:ONEFLEETNODE\AzureHost-VMService-Sev-3-4
76195:XSTORE\XCompute
51564:CLOUDSERVERINFRASTRUCTURECSI\CSIStorage
65292:ONEFLEETNODE\AzureHost-VMService-Sev-1-2
49975:ONEFLEETNODE\AzureHost-Agent-Sev-1-2
11194:XSTORE\FE
51301:RDOS\OSBasePlatform
102763:RDOS\zHYPSMELOW(HYPSMEuseonly)
67106:RDOS\AzureHostOSDRINinjas-Mv2Investigations
11197:XSTORE\Stream
58608:AZURERT\GuestAgent
83383:ONEFLEETNODE\TeamElixir_LowSev
32087:WINDOWSCLOUDSERVERDEVELOPMENT\CSICMServiceSWandCMFW/BIOS
102826:RDOS\zHYPSMESVX-VST(HYPSMEuseonly)
10426:XSTORE\Triage
51304:RDOS\OSBaseStorage
26151:AZURECOMPUTEINSIGHTS\VMAvailability
76837:ONEFLEETNODE\Aplat_HighSev
22393:AZUREWATSON\Triage
36979:AZUREDIRECTDRIVE\Triage
11199:XSTORE\TableServer
25092:CORELINUXOS\Triage
42293:XSTORE\SustainabilityEngineering
62378:CLOUDNET\NMAgent
33736:XSTORE\Triage-Manual
54452:ONEDEPLOY\PFClientHot
56159:XSTORE\EngineeringSystems
98068:CLOUDHARDWAREINFRASTRUCTUREENGINEERINGCHIE\CHIEFleetEngineeringDebugForum
40844:XSTORE\LargeStorageAccountApprovals
10553:WINDOWSAZUREOPERATIONSCENTER\WASU
32086:WINDOWSCLOUDSERVERDEVELOPMENT\CSIMBBIOS/BMCSupport
58170:RDOS\AzureHostOSDRITraining
11427:CLOUDNET\PhysicalNetworkNE
34141:AZUREPROFILER\IncidentManager
68078:CLOUDSERVERINFRASTRUCTURECSI\CHIEMemory
102827:RDOS\zHYPSMESVX-Device(HYPSMEuseonly)
77939:WSDCFE\ENS\FUN-CloudInfrastructure
51300:RDOS\OSBaseKernel
11196:XSTORE\LocationService
50029:ONEFLEETNODE\NodeServicePrivate
68814:AZURERT\CLSTS
56919:WSDCFE\HCCompute-GuestOSHealth
83776:RDOS\HostOSDeployment
68135:ONEFLEETNODE\AnvilLowSev(NoOnCallRotation)
44482:CLOUDNET\PhysicalNetworkHMO
28556:AZURESTACK\ComputeResourceProvider
51299:RDOS\OSCoreNetworking
58246:HARDWAREHEALTHSERVICE\Triage
94542:RDOS\UnhealthyNodePepsiCo
98735:AZURERT\VMExtBHS
44795:CLOUDSERVERINFRASTRUCTURECSI\CSIDiagnostics
77989:HOSTNETWORKING\HostSDNStack
35654:CLOUDNET\NetworkingNinjas
32106:WINDOWSCLOUDSERVERDEVELOPMENT\CSICMService
84381:CLOUDSERVERINFRASTRUCTURECSI\HWReliability
32078:WINDOWSCLOUDSERVERDEVELOPMENT\CSIFPGA
61937:SIGMAOVERLAKESERVICES\Triage
87404:CLOUDHARDWAREINFRASTRUCTUREENGINEERINGCHIE\CHIEEE/Thermal/MechanicalMainRotation
22772:AZURECOMPUTEINSIGHTS\Triage
50979:ONEFLEETNODE\AzureHost-NewTech-Sev-3-4
23739:RDOS\GuestOSDeployments
26161:XSTORE\XInvestigator
80369:COMPUTEMANAGER\AzKepler
87029:AZUREHARDWAREDATACENTERMANAGER\TitanPlatform_LowSev
94816:AZUREHOSTSTORAGE\StorageClientAlerts(Sev34only)
99749:ONEDEPLOY\PFClientLinux
59421:ONEFLEETNODE\ExperimentalAlerts
77944:COMPUTEMANAGER\Scheduler(Allocator)LowPriRotation(Sev3/4)
105567:WSDCFE\CloudOSTriage
11791:AZUREHARDWARE\DatacenterManager
64992:ONEDEPLOY\RepairDetectorDRI
99538:AZURECOMPUTEINSIGHTS\HostHealth-LowMemoryTriage
79765:AZQUALIFY\AzQualifyDataScience
61261:LSGOVERLAKE\Triage
78179:CLOUDNET\PhynetAlerting
87036:AZUREHARDWAREDATACENTERMANAGER\Titan_Agents_LowSev
65445:AZUREHARDWAREDATACENTERMANAGER\FortKnox
87011:WINDOWSPLATFORM\CoreNetworking-NetworkVirtualization
38916:CLOUDNET\SDNDeployment"""
    lines =table.split('\n')
    data_dict = {}
    for line in lines:
        if line:
            key, value = line.split(':')
            data_dict[key] = value
    print(data_dict)
    return data_dict[str(Id)]