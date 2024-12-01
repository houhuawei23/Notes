```bash
groups <username> # to check groups of a user
usermod -G <groupname> <username> # to add a user to a group
usermod -G <groupname> -d <directory> <username> # to change the home directory of a user
usermod -s <shell> <username> # to change the shell of a user

# give user sudo access
sudo vim /etc/sudoers
<username> ALL=(ALL) ALL # add the line above to the end of the file

# add user to sudo|wheel group
sudo usermod -aG sudo <username>
```

```bash
su

Switch shell to another user.
More information: https://manned.org/su.

 - Switch to superuser (requires the root password):
   su

 - Switch to a given user (requires the user's password):
   su username

 - Switch to a given user and simulate a full login shell:
   su - username

 - Execute a command as another user:
   su - username -c "command"
```

```bash
adduser

User addition utility.
More information: https://manned.org/adduser.

 - Create a new user with a default home directory and prompt the user to set a password:
   adduser username

 - Create a new user without a home directory:
   adduser --no-create-home username

 - Create a new user with a home directory at the specified path:
   adduser --home path/to/home username

 - Create a new user with the specified shell set as the login shell:
   adduser --shell path/to/shell username

 - Create a new user belonging to the specified group:
   adduser --ingroup group username
```

```bash
users

Display a list of logged in users.
See also: useradd, userdel, usermod.
More information: https://www.gnu.org/software/coreutils/users.

 - Print logged in usernames:
   users

 - Print logged in usernames according to a given file:
   users /var/log/wmtp

```

```bash
usermod

Modify a user account.
See also: users, useradd, userdel.
More information: https://manned.org/usermod.

 - Change a username:
   sudo usermod -l|--login new_username username

 - Change a user ID:
   sudo usermod -u|--uid id username

 - Change a user shell:
   sudo usermod -s|--shell path/to/shell username

 - Add a user to supplementary groups (mind the lack of whitespace):
   sudo usermod -a|--append -G|--groups group1,group2,... username

 - Change a user home directory:
   sudo usermod -m|--move-home -d|--home path/to/new_home username

```


```bash
gpasswd

Administer /etc/group and /etc/gshadow.
More information: https://manned.org/gpasswd.

 - Define group administrators:
   sudo gpasswd -A user1,user2 group

 - Set the list of group members:
   sudo gpasswd -M user1,user2 group

 - Create a password for the named group:
   gpasswd group

 - Add a user to the named group:
   gpasswd -a user group

 - Remove a user from the named group:
   gpasswd -d user group
   
(base) ➜  ~ gpasswd                                 
Usage: gpasswd [option] GROUP

Options:
  -a, --add USER                add USER to GROUP
  -d, --delete USER             remove USER from GROUP
  -h, --help                    display this help message and exit
  -Q, --root CHROOT_DIR         directory to chroot into
  -r, --remove-password         remove the GROUP's password
  -R, --restrict                restrict access to GROUP to its members
  -M, --members USER,...        set the list of members of GROUP
  -A, --administrators ADMIN,...
                                set the list of administrators for GROUP
Except for the -A and -M options, the options cannot be combined.

```