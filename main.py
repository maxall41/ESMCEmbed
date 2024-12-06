import fire
import h5py
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm


def main(fasta_file, model="esmc_600m", output_file="output.h5py"):
    client = ESMC.from_pretrained(model).to("cuda")  # or "cpu"
    with h5py.File(output_file, "w") as h5f:
        for seq_record in tqdm(SeqIO.parse(fasta_file, "fasta")):
            protein = ESMProtein(sequence=seq_record.seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )

            h5f.create_dataset(
                f"{seq_record.id}_{model}",
                data=logits_output.embeddings.cpu().numpy(),
            )


if __name__ == "__main__":
    fire.Fire(main)
