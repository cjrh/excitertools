# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-2-Clause

import os, sys
self_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(self_dir, ".."))

from conf_common import *

master_doc = os.path.basename(self_dir).upper()
