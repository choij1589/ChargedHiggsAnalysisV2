#!/bin/bash
tar -c python | pigz -p 8 > archive/python.tar.gz
