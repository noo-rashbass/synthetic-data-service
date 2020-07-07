# synthetic-data-service
A blank repository for the HDI 2020 intern work

# Using this repo
1. Clone it (you probably already have if you're reading this)
  - If not paste `git clone https://github.com/noo-rashbass/synthetic-data-service.git` into terminal
2. This is just one repo however it may be more helpful to have a seperate ones for Engineering and Data work
3. The general git practice, at least in my opinion is 
	- `git pull` - update your changes to the upstream
	- `git checkout -b <the name of your branch>` - creates a new branch, this allows for the organisation of your changes whilst the master branch can be updated independently
	- make your changes - make any changes to the code on your local copy
	- `git commit -m 'your commit message'` - make git log your changes
		- n.b. this will probably need you to run `git add *` or `git add path/to/specific/file`
	- `git checkout master` - move back to master, your changes aren't here yet, but don't worry they're still on your branch
	- `git pull` - update master to any upstream changes, so your changes are added to the most up-to-date version
	- `git merge <the name of your branch` - combine your branch changes with those in the repository
	- optional: `git branch -d <the name of your branch>` - delete the branch if you're done with it
	- `git push` - send your updated local copy to the upstream repo
