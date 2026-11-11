#!/usr/bin/env python3
"""Safely stage 2026-10-27 evidence on a NEW git branch. Never change main."""
import argparse, pathlib, subprocess, shutil, sys
BRANCH='stage/prd32-20261027-closeout'
SOURCE=pathlib.Path(__file__).resolve().parent/'stage_delivery'/'prd32_20261027'
DEST=pathlib.Path('stage_delivery/prd32_20261027')
def git(repo,*args):
 p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(f'git {args}: {p.stderr.strip()}')
 return p.stdout.strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);g=ap.add_mutually_exclusive_group(required=True)
 g.add_argument('--dry-run',action='store_true');g.add_argument('--apply',action='store_true');g.add_argument('--push',action='store_true'); args=ap.parse_args()
 repo=pathlib.Path(args.repo).resolve()
 if not (repo/'.git').exists():raise SystemExit('Not a local git repo: '+str(repo))
 print('Repository:',repo,'Current branch:',git(repo,'branch','--show-current'))
 if args.push:
  if git(repo,'branch','--show-current')!=BRANCH:raise SystemExit('Refusing push: not on target stage branch')
  git(repo,'push','-u','origin',BRANCH);print('PUSHED',BRANCH);return
 if git(repo,'status','--porcelain'):raise SystemExit('STOP: working tree is not clean. Commit or stash existing changes first.')
 # Use a new branch name only if it does not already exist.
 chk=subprocess.run(['git','-C',str(repo),'show-ref','--verify','--quiet','refs/heads/'+BRANCH])
 if chk.returncode==0:raise SystemExit('STOP: stage branch already exists; inspect it manually to avoid duplicate commits.')
 existing=repo/DEST
 if existing.exists():raise SystemExit('STOP: target archive directory already exists; refusing overwrite')
 print('Would add',len(list(SOURCE.rglob('*'))),'path entries under',DEST)
 if args.dry_run:return
 if git(repo,'branch','--show-current')!='main':raise SystemExit('STOP: start from main branch for consistent base')
 git(repo,'switch','-c',BRANCH)
 existing.parent.mkdir(parents=True,exist_ok=True)
 shutil.copytree(SOURCE,existing)
 git(repo,'add','--',str(DEST))
 git(repo,'commit','-m','docs(prd32): archive 2026-10-27 diagnostic evidence')
 print('LOCAL COMMIT CREATED:',git(repo,'rev-parse','--short','HEAD'))
 print('Now run --push if credentials are available. Main remains unchanged.')
if __name__=='__main__':main()
