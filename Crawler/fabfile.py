from __future__ import with_statement
from fabric.api import *

env.hosts = ['ec2-54-212-13-213.us-west-2.compute.amazonaws.com',
             'ec2-54-191-89-62.us-west-2.compute.amazonaws.com']

env.user = 'ec2-user'
env.key_filename = ['/Users/funktor/stokastik.pem','/Users/funktor/stokastik.pem',]


@parallel
def run_df():
    run('df -h')


@parallel
def install_packages_libraries():
    sudo('yum update')
    sudo('yum install python36 python36-pip')
    run('curl -O https://bootstrap.pypa.io/get-pip.py')
    run('python3 get-pip.py --user')
    run('pip3 install fake_useragent')
    run('pip3 install numpy')
    run('pip3 install beautifulsoup4')
    run('pip3 install pandas')
    run('pip3 install requests')
    run('pip3 install lxml')
    run('pip3 install cassandra-driver')
    sudo('yum install git')
    run('git clone https://github.com/funktor/stokastik.git')


@parallel
def install_redis_cli():
    sudo('yum --enablerepo=epel install redis')
    run('pip3 install redis')
    run('pip3 install redis-py-cluster')
    run('pip3 install redlock-py')


@parallel
def git_pull():
    with cd('/home/ec2-user/stokastik'):
        run('git stash')
        run('git pull origin master')


@parallel
def run_crawler():
    with cd('/home/ec2-user/stokastik/Crawler/'):
        run('python3 wiki_crawler.py')

@parallel
def run_amzn_crawler():
    with cd('/home/ec2-user/stokastik/Crawler/'):
        run('python3 amazon_search_description_crawler.py')