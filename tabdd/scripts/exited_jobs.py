import subprocess

def get_full_job_name(job_id):
    result = subprocess.run(["jbinfo", job_id, "-long-long"], stdout=subprocess.PIPE).stdout.decode().strip()
    if not "Job Name" in result: return None
    return result.split("Job Name <")[1].split(">")[0]

result = subprocess.run(["jbinfo", "-a"], stdout=subprocess.PIPE)

jobs = [line.split() for line in result.stdout.decode().strip().split("Finished jobs:\n")[1].split("\n")[1:]]

exited_jobs = [get_full_job_name(j[0]) for j in jobs if j[2] == "EXIT"]
exited_jobs = [j for j in exited_jobs if j]
finished_jobs = [get_full_job_name(j[0]) for j in jobs if j[2] == "DONE"]
finished_jobs = [j for j in finished_jobs if j]

print("These jobs need to run again\n")
for jname in sorted(set(exited_jobs) - set(finished_jobs)):
    print(jname)
