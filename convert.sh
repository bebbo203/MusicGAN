#!/bin/bash

timidity -Ow -o - test.mid | lame - tester.mp3
