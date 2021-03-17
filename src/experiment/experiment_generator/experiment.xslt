<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
                xmlns:exp="https://jcbpyle.github.io/XMMLExperiment">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<!--Main template-->
<xsl:template match="/">
# Copyright 2019 The University of Sheffield
# Author: James Pyle
# Contact: jcbpyle1@sheffield.ac.uk
# Template experiment script file for FLAME GPU agent-based model
#
# University of Sheffield retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# University of Sheffield is strictly prohibited.
#
# For terms of licence agreement please attached licence or view licence 
# on www.flamegpu.com website.
#

<xsl:if test="exp:Experimentation/xmml:Imports"><xsl:for-each select="exp:Experimentation/xmml:Imports/xmml:Import"><xsl:if test="xmml:From">from <xsl:value-of select="xmml:From" /><xsl:text>&#x20;</xsl:text></xsl:if>import <xsl:value-of select="xmml:Module" /><xsl:text>&#xa;</xsl:text></xsl:for-each></xsl:if>
import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY
GPUS_AVAILABLE = cuda.Device(0).count()
OS_NAME = os.name
<xsl:if test="exp:Experimentation/xmml:InitialStates">
PROJECT_DIRECTORY = BASE_DIRECTORY+"<xsl:value-of select="@baseDirectory"/>/"
#InitialStates
<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFile">
initial_state_files = []
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFile">
<xsl:if test="xmml:FileName">
initial_state_files += ["/<xsl:value-of select="xmml:Location"/>/<xsl:value-of select="xmml:FileName"/>.xml"]
</xsl:if>
</xsl:for-each>
</xsl:if>
<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
#Initial state generator function to be created by the user
def initial_state_generator_function_<xsl:value-of select="xmml:FunctionName"/>():

	return<xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
#Generate initial states based on defined ranges/lists/values for all global and agent population variables<xsl:if test="xmml:GeneratorName"> for experiment <xsl:value-of select="xmml:GeneratorName"/></xsl:if>.
def generate_initial_states<xsl:if test="xmml:GeneratorName">_<xsl:value-of select="xmml:GeneratorName"/></xsl:if>(location_name=''):
	global_data = []
	agent_data = []
	vary_per_agent = []
	<xsl:if test="xmml:Globals">
	global_data = {<xsl:for-each select="xmml:Globals/xmml:Global">"<xsl:value-of select="xmml:Name"/>":<xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:when><xsl:when test="xmml:Value/xmml:List"><xsl:choose><xsl:when test="xmml:Value/xmml:List/xmml:Select">random.choices([<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>],k=<xsl:value-of select="xmml:Value/xmml:List/xmml:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in <xsl:if test="xmml:Type='float'">np.a</xsl:if>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise><xsl:if test="xmml:Type='float'">[round(x,6) for x in np.a</xsl:if>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)<xsl:if test="xmml:Type='float'">]</xsl:if></xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:PythonRandom">[<xsl:if test="xmml:Type"><xsl:value-of select="xmml:Type"/>(</xsl:if>random.<xsl:value-of select="xmml:Value/xmml:PythonRandom/xmml:Function"/>(<xsl:value-of select="xmml:Value/xmml:PythonRandom/xmml:Arguments"/>)<xsl:if test="xmml:Type">)</xsl:if>]</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="xmml:Globals/xmml:Global">
	<xsl:if test="xmml:Value">global_data += [["<xsl:value-of select="xmml:Name"/>", <xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue"><xsl:value-of select="xmml:Value/xmml:FixedValue"/></xsl:when><xsl:when test="xmml:Value/xmml:List"><xsl:choose><xsl:when test="xmml:Value/xmml:List/xmml:Select">random.choices([<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>],k=<xsl:value-of select="xmml:Value/xmml:List/xmml:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:PythonRandom">random.<xsl:value-of select="xmml:Value/xmml:PythonRandom/xmml:Function"/>(<xsl:value-of select="xmml:Value/xmml:PythonRandom/xmml:Arguments"/>)</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	</xsl:if>
	<xsl:if test="xmml:Populations">
	<xsl:for-each select="xmml:Populations/xmml:Population">
	<xsl:value-of select="xmml:Agent"/> = {<xsl:if test="xmml:InitialPopulationCount">"initial_population":<xsl:choose><xsl:when test="xmml:InitialPopulationCount/xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:FixedValue"/>]</xsl:when><xsl:when test="xmml:InitialPopulationCount/xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Select">[int(random.<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:InitialPopulationCount/xmml:Value/xmml:PythonRandom">[<xsl:if test="xmml:InitialPopulationCount/xmml:Type"><xsl:value-of select="xmml:InitialPopulationCount/xmml:Type"/>(</xsl:if>random.<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:PythonRandom/xmml:Function"/>(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Value/xmml:PythonRandom/xmml:Arguments"/>)<xsl:if test="xmml:InitialPopulationCount/xmml:Type">)</xsl:if>]</xsl:when><xsl:otherwise>[0]</xsl:otherwise></xsl:choose>, </xsl:if><xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:if test="not(xmml:Value/xmml:PerAgentRandom)">"<xsl:value-of select="xmml:Name"/>":<xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in <xsl:if test="xmml:Type='float'">np.a</xsl:if>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise><xsl:if test="xmml:Type='float'">[round(x,6) for x in np.a</xsl:if>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)<xsl:if test="xmml:Type='float'">]</xsl:if></xsl:otherwise></xsl:choose></xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:if></xsl:for-each>}
	<xsl:value-of select="xmml:Agent"/>_vary_per_agent = {<xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:if test="xmml:Value/xmml:PerAgentRandom">"<xsl:value-of select="xmml:Name"/>":[<xsl:value-of select="xmml:Value/xmml:PerAgentRandom/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:PerAgentRandom/xmml:Max"/><xsl:if test="xmml:Value/xmml:PerAgentRandom/xmml:Distribution">,"<xsl:value-of select="xmml:Value/xmml:PerAgentRandom/xmml:Distribution"/>"</xsl:if><xsl:if test="xmml:Type">,<xsl:value-of select="xmml:Type"/></xsl:if>],</xsl:if></xsl:for-each>}<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text>
	</xsl:for-each>
	agent_data = {<xsl:for-each select="xmml:Populations/xmml:Population">"<xsl:value-of select="xmml:Agent"/>":<xsl:value-of select="xmml:Agent"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="xmml:Populations/xmml:Population">
	<xsl:if test="xmml:Agent">agent_data += [["<xsl:value-of select="xmml:Agent"/>",["initial_population",<xsl:if test="xmml:InitialPopulationCount/xmml:FixedValue">[<xsl:value-of select="xmml:InitialPopulationCount/xmml:FixedValue"/>]</xsl:if><xsl:if test="xmml:InitialPopulationCount/xmml:Range"><xsl:choose><xsl:when test="xmml:InitialPopulationCount/xmml:Range/xmml:Select">[int(random.<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/><xsl:if test="xmml:InitialPopulationCount/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>],<xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text>["<xsl:value-of select="xmml:Name"/>",<xsl:if test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:if><xsl:if test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>]<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	vary_per_agent = {<xsl:for-each select="xmml:Populations/xmml:Population">"<xsl:value-of select="xmml:Agent"/>":<xsl:value-of select="xmml:Agent"/>_vary_per_agent<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}

	</xsl:if>
	prefix_components = []
	prefix = ''
	<xsl:if test="xmml:Files/xmml:Prefix">
	<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:AltersWith">prefix_components += [["<xsl:value-of select="text()"/>",global_data["<xsl:value-of select="text()"/>"][0] if len(global_data)>0 else "NA"]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components += [["<xsl:value-of select="xmml:Variable/xmml:Name"/>", <xsl:if test="xmml:Variable/xmml:Type = 'str'">"</xsl:if><xsl:value-of select="xmml:Variable/xmml:Initial"/><xsl:if test="xmml:Variable/xmml:Type = 'str'">"</xsl:if>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = <xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
	</xsl:if>
	prefix = location_name+prefix
	if len(global_data)>0:
		global_names = [x for x in global_data]
		unnamed_global_combinations = list(itertools.product(*[y for x,y in global_data.items()]))
		global_combinations = list(zip([global_names for x in range(len(unnamed_global_combinations))],unnamed_global_combinations))
	if len(agent_data)>0:
		agent_names = [x for x in agent_data]
		unnamed_agent_combinations = list(itertools.product(*[z for x,y in agent_data.items() for w,z in y.items()]))
		loc = 0
		agent_combinations = [[] for x in range(len(unnamed_agent_combinations))]
		for an in agent_names:
			num_vars = loc+len(agent_data[an])
			var_names = [x for x in agent_data[an]]
			sublists = [x[loc:num_vars] for x in unnamed_agent_combinations]
			named_combinations = list(zip([var_names for x in range(len(sublists))],sublists))
			for i in range(len(named_combinations)):
				temp_list = [an]
				temp_list += [[named_combinations[i][0][x],[named_combinations[i][1][x]]] for x in range(len(named_combinations[i][0]))]
				agent_combinations[i] += [temp_list]
			loc = num_vars
	if len(global_combinations)>0 and len(agent_combinations)>0:
		for g in global_combinations:
			for a in agent_combinations:
				current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
				<xsl:choose>
				<xsl:when test="xmml:Files/xmml:Prefix">
				prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
				<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
				prefix_strings = [str(y) for x in prefix_components for y in x]
				prefix = location_name+<xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
				initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)
				</xsl:when>
				<xsl:otherwise>initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
				</xsl:choose>
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			<xsl:choose>
			<xsl:when test="xmml:Files/xmml:Prefix">
			prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
			<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = location_name+<xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)
			</xsl:when>
			<xsl:otherwise>initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			<xsl:choose>
			<xsl:when test="xmml:Files/xmml:Prefix">
			prefix_components = [x if not x[0] in a else [x[0],a.index(x[0])[1]] for x in prefix_components]
			<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = location_name+<xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",global_data,current_agent_data)
			</xsl:when>
			<xsl:otherwise>initial_state(<xsl:choose><xsl:when test="xmml:Files/xmml:Location">"<xsl:value-of select="xmml:Files/xmml:Location"/>/",</xsl:when><xsl:otherwise>'',</xsl:otherwise></xsl:choose>str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",global_data,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	else:
		print("No global or agent variations specified for experimentation\n")
	return global_data,agent_data,location_name
</xsl:for-each>
</xsl:if>
#Initial state file creation.
def initial_state(save_location,folder_prefix,file_name,global_information,agent_information):
	#if not os.path.exists(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>"):
		#os.mkdir(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>")
	#SAVE_DIRECTORY = PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>"+"/"
	save_split = save_location.split("/")
	temp = ''
	for i in save_split:
		temp += i+"/"
		if not os.path.exists(PROJECT_DIRECTORY+temp):
			os.mkdir(PROJECT_DIRECTORY+temp)
	fp_split = folder_prefix.split("/")
	temp = ''
	for i in fp_split:
		temp += i+"/"
		if not os.path.exists(PROJECT_DIRECTORY+save_location+temp):
			os.mkdir(PROJECT_DIRECTORY+save_location+temp)
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"/"+save_location+"/"+folder_prefix+"/"
	print("Creating initial state in",SAVE_DIRECTORY,"/",file_name,"\n")
	initial_state_file = open(SAVE_DIRECTORY+str(file_name),"w")
	initial_state_file.write("&lt;states&gt;\n&lt;itno&gt;0&lt;/itno&gt;\n&lt;environment&gt;\n")
	if len(global_information)>0:
		for g in range(len(global_information[0])):
			initial_state_file.write("&lt;"+str(global_information[0][g])+"&gt;"+str(global_information[1][g])+"&lt;/"+str(global_information[0][g])+"&gt;\n")
	initial_state_file.write("&lt;/environment&gt;\n")
	if len(agent_information)>0:
		for i in range(len(agent_information)):
			try:
				ind = [x[0] for x in agent_information[i]].index("initial_population")
			except:
				ind = 0
			num_agents = int(agent_information[i][ind][1][0])
			agent_id = 1
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("&lt;xagent&gt;\n")
				initial_state_file.write("&lt;name&gt;"+str(agent_name)+"&lt;/name&gt;\n")
				initial_state_file.write("&lt;id&gt;"+str(agent_id)+"&lt;/id&gt;\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==4:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(k[1][3](random_method(k[1][0],k[1][1])))+"&lt;/"+str(k[0])+"&gt;\n")
							elif len(k[1])==3:
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(k[1][2](random.uniform(k[1][0],k[1][1])))+"&lt;/"+str(k[0])+"&gt;\n")
							else:
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(random.uniform(k[1][0],k[1][1]))+"&lt;/"+str(k[0])+"&gt;\n")
						elif type(k[1][0])==type(int()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(int(k[1][0]))+"&lt;/"+str(k[0])+"&gt;\n")
						elif type(k[1][0])==type(float()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(float(k[1][0]))+"&lt;/"+str(k[0])+"&gt;\n")
						
				initial_state_file.write("&lt;/xagent&gt;\n")
				agent_id += 1
	initial_state_file.write("&lt;/states&gt;")
	return
</xsl:if>
<xsl:if test="exp:Experimentation/xmml:ExperimentSet">
#ExperimentSet
<xsl:for-each select="exp:Experimentation/xmml:ExperimentSet/xmml:Experiment">
############## <xsl:if test="xmml:ExperimentName"><xsl:value-of select="xmml:ExperimentName" /></xsl:if> ############
<xsl:if test="xmml:Configuration">
<xsl:if test="xmml:Configuration/xmml:ExperimentFunctions">
<xsl:for-each select="xmml:Configuration/xmml:ExperimentFunctions/xmml:Function">
def <xsl:value-of select="xmml:Name" />(<xsl:for-each select="xmml:Arguments/xmml:Argument"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>):
	<xsl:if test="xmml:GlobalVariables">global <xsl:for-each select="xmml:GlobalVariables/xmml:Global"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,<xsl:text>&#x20;</xsl:text></xsl:if></xsl:for-each><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	<xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/> = None<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each></xsl:if>
	experiment_seed = <xsl:choose><xsl:when test="../../../xmml:Seed"><xsl:value-of select="../../../xmml:Seed"/></xsl:when><xsl:otherwise>random.randrange(sys.maxsize)</xsl:otherwise></xsl:choose>
	random.seed(experiment_seed)
	experiment_start_time = datetime.datetime.now()
	<xsl:if test="../../../xmml:InitialState/xmml:Generator">
	<xsl:variable name="generator_name" select="../../../xmml:InitialState/xmml:Generator"/>
	<xsl:variable name="save_location">
		<xsl:for-each select="../../../../../xmml:InitialStates"><xsl:if test="xmml:InitialStateGenerator/xmml:GeneratorName=$generator_name"><xsl:value-of select="xmml:InitialStateGenerator/xmml:Files/xmml:Location"/></xsl:if></xsl:for-each>
	</xsl:variable>
	if not os.path.exists(PROJECT_DIRECTORY+"/<xsl:value-of select="$save_location"/>/"):
		os.mkdir(PROJECT_DIRECTORY+"/<xsl:value-of select="$save_location"/>/")
	experiment_info_file = open(PROJECT_DIRECTORY+"/<xsl:value-of select="$save_location"/>/experiment_information.csv","w")
	experiment_info_file.write("Experiment,<xsl:value-of select="xmml:Name"/>\nSeed,"+str(experiment_seed)+"\nstart_time,"+str(experiment_start_time)+"\n")
	experiment_info_file.close()
	open(PROJECT_DIRECTORY+"<xsl:value-of select="$save_location"/>/simulation_results.csv","w").close()
	<xsl:choose><xsl:when test="../../xmml:Repeats">
	#Run for desired number of repeats
	REPEATS = <xsl:value-of select="../../xmml:Repeats"/>
	base_output_directory = PROJECT_DIRECTORY+"<xsl:value-of select="$save_location"/>/"
	for i in range(REPEATS):
		location_name = "<xsl:value-of select="xmml:Name" />_"+str(i)+"/"
		generate_initial_states_<xsl:value-of select="../../../xmml:InitialState/xmml:Generator"/>(location_name)
		#generation_time = datetime.datetime.now()
		#ef = open(PROJECT_DIRECTORY+"<xsl:value-of select="$save_location"/>/experiment_information.csv","a")
		#ef.write("repeat,"+str(i)+"\ninitial_states_generated,"+str(generation_time)+"\n")
		#ef.close()
		#Model executable
		executable = ""
		simulation_command = ""
		os_walk = list(os.walk(base_output_directory+location_name))
		if len(os_walk[0][1])>1:
			initial_states = [x[0] for x in os_walk][1:]
		else:
			initial_states = [x[0] for x in os_walk]
		for j in initial_states:
			current_initial_state = j+"/0.xml"
			if OS_NAME=='nt':
				executable = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />.exe"
				simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
			else:
				executable = "./"+PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />"
				simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
			print(simulation_command)
			#Run simulation
			os.system(simulation_command)

			#Parse results
			results_file = open(j+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:FileName"/>","r")
			results = results_file.readlines()
			results_file.close()
			sim_results_file = open(PROJECT_DIRECTORY+"<xsl:value-of select="$save_location"/>/simulation_results.csv","a")
			for res in results:
				sim_results_file.write(res)
			sim_results_file.write("\n")
			sim_results_file.close()
			print(results)
	experiment_completion_time = datetime.datetime.now()
	time_taken = experiment_completion_time-experiment_start_time
	ef = open(PROJECT_DIRECTORY+"<xsl:value-of select="$save_location"/>/experiment_information.csv","a")
	ef.write("completion_time,"+str(experiment_completion_time)+"\ntime_taken,"+str(time_taken)+"\n")
	</xsl:when><xsl:otherwise>
	<xsl:variable name="file_name">
		<xsl:for-each select="../../../../../xmml:InitialStates"><xsl:if test="xmml:InitialStateGenerator/xmml:GeneratorName=$generator_name"><xsl:value-of select="xmml:InitialStateGenerator/xmml:Files/xmml:InitialFileName"/></xsl:if></xsl:for-each>
	</xsl:variable>
	g,a,file_loc = generate_initial_states_<xsl:value-of select="$generator_name"/>()
	#Model executable
	executable = ""
	simulation_command = ""
	current_initial_state = file_loc+"/<xsl:value-of select="$save_location"/>/<xsl:value-of select="$file_name"/>"
	if OS_NAME=='nt':
		executable = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />.exe"
		simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
	else:
		executable = "./"+PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />"
		simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
	print(simulation_command)
	#Run simulation
	os.system(simulation_command)

	#Parse results
	results_file = open(j+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:FileName"/>","r")
	results = results_file.readlines()
	results_file.close()
	print(results)
	</xsl:otherwise>
	</xsl:choose>
	</xsl:if>
	<xsl:if test="not(../../../xmml:InitialState/xmml:Generator)">
	#Model executable
	executable = ""
	simulation_command = ""
	os_walk = list(os.walk("../../<xsl:value-of select="../../../../../xmml:InitialStates/xmml:InitialStateFile/xmml:Location"/>"))
	if len(os_walk[0][1])>1:
		initial_states = [x[0] for x in os_walk][1:]
	else:
		initial_states = [x[0] for x in os_walk]
	for i in initial_states:
		current_initial_state = i+"/0.xml"
		if OS_NAME=='nt':
			executable = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />.exe"
			simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		else:
			executable = "./"+PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />"
			simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		print(simulation_command)
		
		
		#Run simulation
		os.system(simulation_command)

		#Parse results
		results_file = open(i+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:FileName"/>","r")
		results = results_file.readlines()
		results_file.close()
		print(results)
		</xsl:if>
	return <xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,<xsl:text>&#x20;</xsl:text></xsl:if></xsl:for-each></xsl:if><xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:if>
<xsl:if test="not(xmml:Configuration/xmml:ExperimentFunctions) and xmml:Configuration"><xsl:if test="xmml:Configuration/xmml:Repeats">
#Run for desired number of repeats
REPEATS = <xsl:value-of select="xmml:Configuration/xmml:Repeats"/>
for i in range(REPEATS):
	</xsl:if>initial_state_creation_<xsl:value-of select="xmml:InitialState/xmml:Generator"/>(file_name,base_agent_information)
	Run simulation
	os.system(simulation_command)
	Parse results
	results_file = open("../../<xsl:value-of select="xmml:SimulationOutput/xmml:Location"/>"+INSERT_FILE_DIRECTORY_AND_NAME_HERE,"r")
	results = results_file.readlines()
	results_file.close()
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:if>
def main():
	<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
	<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
	#Initial state generator function to be created by the user
	initial_state_generator_function_<xsl:value-of select="xmml:FunctionName"/>()
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
	#Initial state creation function
	#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

	#Generation functions (will automatically call initial state generation function)
	<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
	#generate_initial_states<xsl:if test="xmml:GeneratorName">_<xsl:value-of select="xmml:GeneratorName"/></xsl:if>()
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="exp:Experimentation/xmml:ExperimentSet">
	#Experiment Set user defined functions
	<xsl:for-each select="exp:Experimentation/xmml:ExperimentSet/xmml:Experiment">
	<xsl:if test="xmml:Configuration/xmml:ExperimentVariables">
	<xsl:for-each select="xmml:Configuration/xmml:ExperimentVariables/xmml:Variable">
	<xsl:value-of select="xmml:Name" /> = <xsl:if test="not(xmml:Type='tuple')"><xsl:value-of select="xmml:Type" /></xsl:if>(<xsl:value-of select="xmml:Value" />)
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="xmml:Configuration/xmml:ExperimentFunctions">
	<xsl:for-each select="xmml:Configuration/xmml:ExperimentFunctions/xmml:Function">
	<xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if><xsl:text>&#x20;</xsl:text></xsl:for-each>= </xsl:if><xsl:value-of select="xmml:Name" />(<xsl:for-each select="xmml:Arguments/xmml:Argument"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>)
	</xsl:for-each>
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	return

if __name__ == "__main__":
	main()


######################## TEMPLATE SEARCH AND SURROGATE MODELLING CODE ##########################################################

##Template (1+1)GA search
#from deap import base
#from deap import creator
#from deap import tools
#import numpy as np
#import datetime
#import queue
#import threading
#
##Alterable parameters, recommend using larger mu (e.g. 100) to reduce chance of being stuck in local optima and population domination by variations of strong candidate, similar with lambda (e.g. 25)
#mu = 1
#lam = 1
#max_generations = 100
##Maximum run time in minutes
#max_time = 100
#crossover = True
#mutation_rate = 0.2
#mates_rate = 0.5
##Threshold at which a candidate solution is considered optimal
#optimal_fitness = 0.95
##Provide a list with min and max for each parameter 
##parameter_limits = [[parameter1_min,parameter1_max],[parameter2_min,parameter2_max]]
#output_file = "ga_results.csv"
#cwd = os.cwwd()+"/"
#logged_statistics = ["mean", "std", "min", "max"]
#
#def genetic_algorithm(mu,lam,max_generations,max_time,loc,output_file):
#	global curr_pop, statistics, toolbox
#	if not os.path.exists(cwd+"ga_temp/"):
#		os.mkdir(cwd+"ga_temp/")
#	working_directory = cwd+"ga_temp/"
#	if not os.path.exists(working_directory+"optimal_solutions_discovered.csv"):
#		open(working_directory+"optimal_solutions_discovered.csv","w").close()
#	#Create a fitness function +ve for maximisation, -ve for minimisation
#	creator.create("Fitness",base.Fitness,weights=(1.0,))#ALTER THIS FITNESS WEIGHTING (possible to have multiple weightings for multiple values e.g. minimise a and c but maximise b with weightings (-1.0,0.75,-0.5,))
#	creator.create("Individual",list,fitness=creator.Fitness)
#	toolbox = base.Toolbox()
#	toolbox.register("individual",create_individual,creator.Individual)
#	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#	#New statistics should be created for each fitness value to be tracked (and the log method and required globals altered accordingly)
#	statistics = tools.Statistics(lambda individual: individual.fitness.values[0])
#	for s in logged_statistics:
#		method = getattr(np,s)
#		statistics.register(s,np.method)
#	logbook = tools.Logbook()
#	logbook.header = ['generation', 'evaluations'] + (statistics.fields if statistics else [])
#	toolbox.register("select_parents", select_parents)
#	toolbox.register("mutate",mutate)
#	toolbox.register("mate",mate)
#
#	current_generation = 0
#	#Initialise a population of mu individuals
#	population = toolbox.population(n=mu)
#	start_time = datetime.datetime.now()
#	print("Initial population evalauation (Generation 0)")
#	#Evaluate initial population
#	initial_fitnesses = evaluate_population(population)
#	candidates_evaluated = mu
#	#Record results per GA in file named the same the current seed being used for the random module
#	unique_run_seed = random.randrange(sys.maxsize)
#	seed_record = working_directory+str(unique_run_seed)+".csv"
#	if not os.path.exists(seed_record):
#		population_record = open(seed_record,"w")
#	else:
#		population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
#	population_record.write("generation,0,mu,"+str(mu)+",lambda,"+str(lam)+"\n")
#	#Set popualtion fitness to evaluated fitness
#	for i in range(len(initial_fitnesses)):
#		population[i].fitness.values = initial_fitnesses[i][0]
#		population_record.write("\tParameters,")
#		for j in population[i][0].tolist():
#			population_record.write(str(j)+",")
#		population_record.write("Fitness,"+str(population[i].fitness.values)+"\n")
#	population_record.close()
#	#Record initial population in the logbook
#	log(logbook, population, current_generation, mu)
#
#	#Begin generational GA process
#	end_conditions = False
#	optimal_solutions = []
#	optimal_count = 0
#	while(current_generation&lt;max_generations and (not end_conditions)):
#		current_generation += 1
#		print("\t Generation:",current_generation)
#		generational_evaluations = 0
#		curr_pop = 0
#		offspring = []
#		evaluations = []
#		#Generate offspring candidates. If crossover is being used, it is done before mutation
#		for i in range(lam):
#			mate_chance = random.uniform(0,1)
#			if mate_chance&lt;mates_rate and (not crossover):
#				child = toolbox.individual()
#			else:
#				parent1, parent2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
#				child = toolbox.mate(parent1, parent2)
#			offspring += [child]
#		#Mutate new candidates
#		for off in offspring:
#			off, = toolbox.mutate(off)
#		generational_evaluations += len(offspring)
#		evaluations = evaluate_population(offspring)
#		for i in range(len(evaluations)):
#			offspring[i].fitness.values = evaluations[i][0]
#		candidates_evaluated += generational_evaluations
#		#Select the next generation, favouring the offspring in the event of equal fitness values
#		population, new_individuals = favour_offspring(population, offspring, mu)
#		#Print a report about the current generation
#		if generational_evaluations&gt;0:
#			log(logbook, population, current_generation, generational_evaluations)
#		#Save to file in case of early exit
#		#log_fitness = open(working_directory+"current_ga_fitness_log.csv","w")
#		#log_fitness.write(str(logbook)+"\n")
#		#log_fitness.close()
#		if not os.path.exists(seed_record):
#			population_record = open(seed_record,"w")
#		else:
#			population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
#		check_nonunique = []
#		for p in population:
#			population_record.write("\t")
#			for q in p[0].tolist():
#				population_record.write(str(q)+",")
#			population_record.write("fitness,"+str(p.fitness.values)+",fitnesses,"+str(p.fitnesses.values)+"\n")
#			if p.fitness.values[0]&gt;optimal_fitness:
#				for opt in optimal_solutions:
#					check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
#				if not any(check_nonunique):
#					optimal_solutions.append((p,current_generation))
#		population_record.write("SimulationGA,generation,"+str(current_generation)+"\n")
#		population_record.close()
#		end_time = datetime.datetime.now()
#		time_taken = end_time-start_time
#		opti = optimal_solutions[optimal_count:]
#		if len(opti)>0:
#			opt = open(working_directory+"optimal_solutions_discovered.csv","a")
#			for b in opti:
#				opt.write("SimulationGA,"+str(unique_run_seed)+",Solution_Parameters,"+str(b[0][0].tolist())+",Fitness,"+str(b[0].fitness.values)+",Discovered_Generation,"+str(b[1])+",Discovered_Time,"+str(end_time)+"\n")
#			opt.close()
#		optimal_count = len(optimal_solutions)
#
#	#Record GA results
#	if not os.path.exists(cwd+output_file):
#		results_file = open(cwd+output_file,"w")
#	results_file = open(cwd+results_file,"a")
#	results_file.write(str(logbook)+"\n")
#	results_file.close()
#	if not os.path.exists(cwd+"ga_times.csv"):
#		open(loc+"times.csv","w").close()
#	time = open(loc+"times.csv","a")
#	time.write("ga_seed,"+str(unique_run_seed)+",started_at,"+str(start_time)+",ended_at,"+str(end_time)+",total_time,"+str(time_taken)+"\n")
#	time.close()
#	return
#
#def create_individual(container):
#	global curr_pop, parameter_limits
#	new = [0]*(len(parameter_limits)+1)
#	for i in range(len(parameter_limits)):
#		if type(parameter_limits[i][0])==type(int()):
#			new[i] = int(random.uniform(parameter_limits[i][0], parameter_limits[i][1]))
#		else:
#			new[i] = round(random.uniform(parameter_limits[i][0], parameter_limits[i][1]),6)
#	new[-1] = curr_pop
#	curr_pop += 1
#	new = np.array(new, dtype=np.float64).reshape(1,-1)
#	return container(new)
#
#def favour_offspring(parents, offspring, MU):
#	choice = (list(zip(parents, [0]*len(parents))) +
#				list(zip(offspring, [1]*len(offspring))))
#	choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
#	return [x[0] for x in choice[:MU]], [x[0] for x in choice[:MU] if x[1]==1]
#
#def log(logbook, population, gen, evals):
#	global statistics
#	record = statistics.compile(population) if statistics else {}
#	logbook.record(generation=gen,evaluations=evals,**record)
#	return
#
#def evaluate_population(population):
#	evaluated_population = population
#	return evaluated_population
#
##Define a function for crossover between 2 individuals (many are available in deap if individuals are in bitstring form)
#def mate(parent1, parent2):
#	global toolbox
#	child = toolbox.individual()
#	return child
#
##Define a function for mutating an individual (many are available in deap if individuals are in bitstring form)
#def mutate(individual):
#	global toolbox
#
#	return individual,
#
##Define a function for selecting parents (many are available in deap)
#def select_parents(individuals,k):
#	global toolbox
#	#Example selection function, randomly select 2 parents from population
#	#parents = [random.choice(individuals) for i in range(k)]
#	#return [toolbox.clone(ind) for ind in parents]
#
</xsl:template>
</xsl:stylesheet>