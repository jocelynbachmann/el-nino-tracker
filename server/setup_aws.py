"""One-time setup: save the team's AWS key on THIS machine so the server can read from S3.

    python setup_aws.py

Same thing as Step 1 of the hub's starter notebook. The key is stored privately in
~/.aws/credentials under the profile "hack-team" - never in this repo.
"""
import configparser
import getpass
import os

cred_p = os.path.expanduser("~/.aws/credentials")
cfg_p = os.path.expanduser("~/.aws/config")
os.makedirs(os.path.dirname(cred_p), exist_ok=True)

cred = configparser.ConfigParser()
cred.read(cred_p)
if "hack-team" in cred:
    print("A hack-team key is already saved. (To replace it, remove that section from ~/.aws/credentials.)")
else:
    cred["hack-team"] = {
        "aws_access_key_id": getpass.getpass("Access key ID: ").strip(),
        "aws_secret_access_key": getpass.getpass("Secret access key: ").strip(),
    }
    with open(cred_p, "w") as f:
        cred.write(f)
    os.chmod(cred_p, 0o600)
    cfg = configparser.ConfigParser()
    cfg.read(cfg_p)
    cfg["profile hack-team"] = {"region": "us-west-2"}
    with open(cfg_p, "w") as f:
        cfg.write(f)
    print("Saved.")

try:
    import s3fs
    files = s3fs.S3FileSystem(profile="hack-team").ls("ocean-hackathon-2026/teams/team-elnino/processed")
    print("Connected. Team processed folder contains:", [f.split("/")[-1] for f in files])
except Exception as e:  # noqa: BLE001
    print("Saved, but couldn't list the bucket yet:", e)
