# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import distutils.command.build
from setuptools import setup

description = """CMHA - Counterfactual Memory Hierarchical Agent for StarCraft II

Extends LLM-PySC2 (Apache 2.0) with SPC macro reasoning and Tactical-Phi
micro control. Built on DeepMind's PySC2 API.

Anonymous submission — supplementary code release.
"""


class BuildCommand(distutils.command.build.build):

  def initialize_options(self):
    distutils.command.build.build.initialize_options(self)
    self.build_base = '_build'


setup(
    name='cmha-llm-pysc2',
    version='0.2.0',
    description='CMHA: hierarchical LLM agent for StarCraft II (LLM-PySC2 fork).',
    long_description=description,
    author='Anonymous',
    author_email='',
    cmdclass={'build': BuildCommand},
    license='Apache License, Version 2.0',
    keywords='StarCraft AI',
    url='',
    packages=[
        'pysc2',
        'pysc2.agents',
        'pysc2.bin',
        'pysc2.env',
        'pysc2.lib',
        'pysc2.maps',
        'pysc2.run_configs',
        'pysc2.tests',
        'llm_pysc2',
        'llm_pysc2.agents',
        'llm_pysc2.bin',
        'llm_pysc2.cfg',
        'llm_pysc2.lib',
        'llm_pysc2.lib.action',
        'llm_pysc2.lib.obs',
    ],
    install_requires=[
        'absl-py>=0.1.0',
        'deepdiff',
        'dm_env',
        'enum34',
        'mock',
        'mpyq',
        'numpy>=1.10',
        'portpicker>=1.2.0',
        'protobuf==3.20.0',
        'openai==1.73.0',
        'pygame',
        'requests',
        's2clientprotocol>=4.10.1.75800.0',
        's2protocol',
        'sk-video',
        'websocket-client',
        'loguru',
        'pillow',
        'llamaapi',
        'zhipuai',
        # 'google-generativeai',
        # 'anthropic',
        # 'google',
    ],
    entry_points={
        'console_scripts': [
            'pysc2_agent = pysc2.bin.agent:entry_point',
            'pysc2_play = pysc2.bin.play:entry_point',
            'pysc2_replay_info = pysc2.bin.replay_info:entry_point',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.9',  # llamaapi requires python > 3.9
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
