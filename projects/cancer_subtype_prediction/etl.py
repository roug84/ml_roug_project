import os
import pandas as pd
import wget
import requests

import ftplib


def download_file(url: str, destination: str) -> None:
    """
    Downloads a file from a specified URL to a destination.

    :param url: URL of the file to be downloaded.
    :param destination: Destination path to save the downloaded file.
    """
    wget.download(url, destination)


def extract_gene_names_from_gtf(gtf_file_path: str) -> list:
    """
    Extracts protein-coding gene names from a specified GENCODE GTF file.

    :param gtf_file_path: Path to the GENCODE GTF file.
    :return: List of unique protein-coding gene names.
    """
    gene_names = set()

    with open(gtf_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            columns = line.strip().split('\t')

            if "gene_type \"protein_coding\"" in columns[8]:
                gene_info = columns[8]
                for info in gene_info.split(';'):
                    info = info.strip()
                    if info.startswith("gene_name"):
                        gene_name = info.split('"')[1]
                        gene_names.add(gene_name)
                        break

    return list(gene_names)


def download_and_extract_protein_coding_genes(gencode_release: str, in_path_to_save_df: str
                                              ) -> list:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts
    protein-coding gene names.

    :param gencode_release: GENCODE release version, e.g., "v38".
    :param in_path_to_save_df: Path to save the dataframe containing the gene names.
    :return: List of unique protein-coding gene names.
    """
    if os.path.exists(in_path_to_save_df):
        protein_coding_genes = pd.read_csv(in_path_to_save_df)
    else:
        base_url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{gencode_release}/gencode.v{gencode_release}.annotation.gtf.gz"
        destination = f"gencode.v{gencode_release}.annotation.gtf.gz"

        # Download the file
        print(f"Downloading GENCODE {gencode_release} GTF file...")
        download_file(base_url, destination)

        # Unzipping the file
        os.system(f"gunzip {destination}")

        # Extract gene names
        gtf_file_path = destination.replace(".gz", "")
        genes_list = extract_gene_names_from_gtf(gtf_file_path)
        protein_coding_genes = pd.DataFrame(genes_list, columns=['gene_name'])
        protein_coding_genes.to_csv(in_path_to_save_df)

    return protein_coding_genes


def extract_gene_id_to_name_mapping(in_path_to_save_df: str, gencode_release: str) -> dict:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts a
    mapping from Ensembl gene IDs to gene names.

    :param in_path_to_save_df: Path to save the dataframe containing the gene ID to name mapping.
    :param gencode_release: GENCODE release version, e.g., "v38".
    :return: Dictionary mapping Ensembl gene IDs to gene names.
    """
    if os.path.exists(in_path_to_save_df):
        df = pd.read_csv(in_path_to_save_df)
    else:
        base_url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{gencode_release}/gencode.v{gencode_release}.annotation.gtf.gz"
        destination = f"gencode.v{gencode_release}.annotation.gtf.gz"

        # Download the file
        print(f"Downloading GENCODE {gencode_release} GTF file...")
        download_file(base_url, destination)
        # Unzipping the file
        os.system(f"gunzip {destination}")

        gtf_file_path = destination.replace(".gz", "")
        gene_id_to_name = {}
        with open(gtf_file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if fields[2] == "gene":
                    info = {x.split()[0]: x.split()[1].replace('"', '').replace(';', '') for x in
                            fields[8].split('; ')}
                    if 'gene_type' in info and info['gene_type'] == "protein_coding":
                        gene_id_to_name[info['gene_id']] = info['gene_name']
        df = pd.DataFrame(list(gene_id_to_name.items()), columns=['Gene ID', 'Gene Name'])
        df.to_csv(in_path_to_save_df, index=False)
    return df


def ensembl_gene_to_protein_id(ensembl_gene_id: str) -> str:
    """
    Retrieve Ensembl protein ID for a given Ensembl gene ID.

    :param ensembl_gene_id: Ensembl gene ID to look up.
    :return: Ensembl protein ID if found, otherwise None.
    """
    base_url = "https://rest.ensembl.org"
    endpoint = f"/lookup/id/{ensembl_gene_id}?expand=1"

    response = requests.get(base_url + endpoint, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        data = response.json()
        if "Transcript" in data:
            transcript_info = data["Transcript"]
            for transcript in transcript_info:
                if "Translation" in transcript:
                    protein_id = transcript["Translation"]["id"]
                    return protein_id
    return None


def convert_and_save_mapping(ensembl_gene_id_list: list, in_path: str) -> pd.DataFrame:
    """
    Convert Ensembl gene IDs to protein IDs and save the mapping to a CSV file.

    :param ensembl_gene_id_list: List of Ensembl gene IDs.
    :param in_path: Path where the resulting CSV will be saved.
    :return: DataFrame containing the mapping from Ensembl gene IDs to protein IDs.
    """
    ensembl_gene_to_protein_mapping = {}

    for ensembl_gene_id in ensembl_gene_id_list:
        print(ensembl_gene_id)
        protein_id = ensembl_gene_to_protein_id(ensembl_gene_id.split('.')[0])

        if protein_id:
            print(f"Ensembl Gene ID: {ensembl_gene_id} -> Ensembl Protein ID: {protein_id}")
            ensembl_gene_to_protein_mapping[ensembl_gene_id] = protein_id
        else:
            # print(f"No Ensembl Protein ID found for Ensembl Gene ID: {ensembl_gene_id}")
            ensembl_gene_to_protein_mapping[ensembl_gene_id] = 'Not found'

    mapping_df = pd.DataFrame.from_dict(ensembl_gene_to_protein_mapping, orient='index',
                                        columns=['Ensembl_Protein_ID'])
    mapping_df.to_csv(in_path, index=True)
    return mapping_df

# ENSEMBL to Gene names


def download_file_from_ftp(url: str, file_path: str) -> None:
    """
    Downloads a file from an FTP URL and saves it to the specified file path.
    """
    url_parts = url.split("/")
    server = url_parts[2]
    file_dir = "/".join(url_parts[3:-1])
    filename = url_parts[-1]

    with ftplib.FTP(server) as ftp:
        ftp.login()
        ftp.cwd(file_dir)

        with open(file_path, 'wb') as f:
            ftp.retrbinary(f"RETR {filename}", f.write)


def extract_gene_id_to_name_mapping(in_path_to_save_df: str,
                                    in_gencode_release: int
                                    ) -> pd.DataFrame:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts a
    mapping from Ensembl gene IDs to gene names.

    :param in_path_to_save_df: Path to save the dataframe containing the gene ID to name mapping.
    :param in_gencode_release: GENCODE release version, e.g., "v38".
    :return: DataFrame mapping Ensembl gene IDs to gene names.
    """
    if os.path.exists(in_path_to_save_df):
        df = pd.read_csv(in_path_to_save_df)
    else:
        str_gencode_release = str(in_gencode_release)
        base_url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{str_gencode_release}/gencode.v{str_gencode_release}.annotation.gtf.gz"

        # Specify a directory for storing the GTF files (modify as necessary)
        gtf_dir = "gtf_files"
        os.makedirs(gtf_dir, exist_ok=True)

        destination = os.path.join(gtf_dir, f"gencode.v{str_gencode_release}.annotation.gtf.gz")

        # Download the file
        print(f"Downloading GENCODE {str_gencode_release} GTF file...")
        download_file_from_ftp(base_url, destination)
        # Unzipping the file
        os.system(f"gunzip {destination}")

        gtf_file_path = destination.replace(".gz", "")
        gene_id_to_name = {}
        with open(gtf_file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if fields[2] == "gene":
                    info = {x.split()[0]: x.split()[1].replace('"', '').replace(';', '') for x in
                            fields[8].split('; ')}
                    # Remove the condition to check if gene_type is protein_coding.
                    gene_id_to_name[info['gene_id']] = info['gene_name']

        df = pd.DataFrame(list(gene_id_to_name.items()), columns=['Gene ID', 'Gene Name'])
        df.to_csv(in_path_to_save_df, index=False)

    return df
