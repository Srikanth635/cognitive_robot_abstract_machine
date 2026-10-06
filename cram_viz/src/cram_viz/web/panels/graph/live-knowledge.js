/* Build the Knowledge-tab graph directly from a running plan and bridge world. */
(function (root) {
  'use strict';

  class LiveKnowledgeGraph {
    /** Represent the semantic entities and relations in one live execution. */
    constructor(planSnapshot, bridgeInformation) {
      this.planSnapshot = planSnapshot || { sig: '', nodes: [] };
      this.bridgeInformation = bridgeInformation || { robot: null, objects: [] };
      this.planNodes = this.planSnapshot.nodes || [];
      this.planNodeByIdentifier = {};
      this.planNodes.forEach((node) => {
        this.planNodeByIdentifier[node.id] = node;
      });
      this.actionNodes = this.planNodes.filter((node) => node.kind === 'ActionNode');
      this.nodes = [];
      this.edges = [];
      this.details = {};
      this.addedIdentifiers = new Set();
      this.addedEdges = new Set();
    }

    /** Build a graph-panel payload for the current live execution. */
    build() {
      if (!this.planNodes.length) return this.emptyPayload();
      this.rootPlanNode = this.planNodes.find((node) => node.parent === null);
      this.addPlan();
      this.addRobot();
      this.addObjects();
      this.addActions();
      return {
        ok: true,
        crumb: 'live execution knowledge',
        nodes: this.nodes,
        edges: this.edges,
        details: this.details,
        legend: [
          { group: 'goal', label: 'Live plan' },
          { group: 'event', label: 'Action' },
          { group: 'robot', label: 'Robot / arm' },
          { group: 'object', label: 'World object' },
        ],
        statusLegend: true,
        spacing: 'wide',
        key: 'knowledge-live',
        sig: this.structureSignature(),
        highlight: this.activeIdentifiers(),
      };
    }

    /** Return the empty payload used before a live plan starts. */
    emptyPayload() {
      return {
        ok: true,
        crumb: 'live execution knowledge',
        nodes: [],
        edges: [],
        details: {},
        key: 'knowledge-live',
        sig: '',
        highlight: [],
      };
    }

    /** Add the live plan root. */
    addPlan() {
      const status = this.rootPlanNode ? this.rootPlanNode.status : 'CREATED';
      this.addNode('live-plan', 'live plan', 'goal', [
        'the plan currently executed by CRAM',
        'status: ' + status,
      ], status);
    }

    /** Add the executing robot and its plan relation. */
    addRobot() {
      const robotName = this.bridgeInformation.robot || 'robot';
      this.addNode('live-robot', robotName, 'robot', ['the robot executing the live plan']);
      this.addEdge('live-plan', 'live-robot', 'prop', 'executed by');
    }

    /** Add every loose object published by the live bridge. */
    addObjects() {
      (this.bridgeInformation.objects || []).forEach((objectName) => {
        this.addObject(objectName);
      });
    }

    /** Add all resolved actions and their semantic relations. */
    addActions() {
      const actionsByName = new Map();
      this.actionNodes.forEach((actionNode) => {
        const executions = actionsByName.get(actionNode.label) || [];
        executions.push(actionNode);
        actionsByName.set(actionNode.label, executions);
      });
      actionsByName.forEach((executions) => {
        const actionNode = executions[0];
        const label = this.actionLabel(actionNode.label);
        const identifier = this.actionIdentifier(actionNode);
        const lines = [
          'a live ' + actionNode.label,
          'executions: ' + executions.length,
        ];
        const statuses = Array.from(new Set(executions.map((node) => node.status)));
        lines.push('status: ' + statuses.join(', '));
        this.addNode(identifier, label, 'event', lines, actionNode.status);
        executions.forEach((execution) => {
          const parentAction = this.actionAncestors(execution)[0];
          this.addEdge(
            parentAction ? this.actionIdentifier(parentAction) : 'live-plan',
            identifier,
            'prop',
            parentAction ? 'has subaction' : 'has action',
          );
          this.addActionEntities(execution, identifier);
        });
      });
    }

    /** Add the arm and target connected to an action. */
    addActionEntities(actionNode, actionIdentifier) {
      if (actionNode.arm) {
        const armName = this.armName(actionNode.arm);
        const armIdentifier = 'live-arm:' + armName;
        this.addNode(armIdentifier, armName + ' arm', 'robot', [
          'the arm assigned to this live action',
        ]);
        this.addEdge('live-robot', armIdentifier, 'prop', 'has part');
        this.addEdge(actionIdentifier, armIdentifier, 'prop', 'uses');
      }
      if (actionNode.target) {
        const objectIdentifier = this.addObject(actionNode.target);
        this.addEdge(actionIdentifier, objectIdentifier, 'prop', 'targets');
      }
    }

    /** Add an object if it is not already present and return its identifier. */
    addObject(objectName) {
      const basename = String(objectName).split('/').pop();
      const identifier = 'live-object:' + basename;
      this.addNode(identifier, this.objectLabel(basename), 'object', [
        'an object in the live world',
        'mesh: ' + basename,
      ]);
      return identifier;
    }

    /** Add one graph node without duplicating its identifier. */
    addNode(identifier, label, group, lines, status) {
      if (this.addedIdentifiers.has(identifier)) return;
      const node = {
        id: identifier,
        label: label,
        group: group,
        title: [label].concat(lines).join('\n'),
      };
      if (status) node.status = status;
      this.nodes.push(node);
      this.details[identifier] = { label: label, group: group, lines: lines };
      this.addedIdentifiers.add(identifier);
    }

    /** Add one semantic graph edge. */
    addEdge(source, target, kind, label) {
      if (source === target) return;
      const identifier = [source, target, kind, label].join('::');
      if (this.addedEdges.has(identifier)) return;
      this.edges.push({ from: source, to: target, kind: kind, label: label });
      this.addedEdges.add(identifier);
    }

    /** Return action ancestors from the immediate parent towards the plan root. */
    actionAncestors(actionNode) {
      const ancestors = [];
      let parentIdentifier = actionNode.parent;
      while (parentIdentifier) {
        const parentNode = this.planNodeByIdentifier[parentIdentifier];
        if (!parentNode) break;
        if (parentNode.kind === 'ActionNode') ancestors.push(parentNode);
        parentIdentifier = parentNode.parent;
      }
      return ancestors;
    }

    /** Return the entities belonging to the action currently being executed. */
    activeIdentifiers() {
      if (!this.rootPlanNode || this.rootPlanNode.status !== 'RUNNING') return [];
      const runningActions = this.actionNodes.filter(
        (actionNode) => actionNode.status === 'RUNNING',
      );
      const currentAction = runningActions[runningActions.length - 1];
      const identifiers = new Set(['live-plan', 'live-robot']);
      if (!currentAction) return Array.from(identifiers);
      const activeActions = [currentAction].concat(this.actionAncestors(currentAction));
      activeActions.forEach((actionNode) => {
        identifiers.add(this.actionIdentifier(actionNode));
        if (actionNode.arm) identifiers.add('live-arm:' + this.armName(actionNode.arm));
        if (actionNode.target) {
          const basename = String(actionNode.target).split('/').pop();
          identifiers.add('live-object:' + basename);
        }
      });
      return Array.from(identifiers);
    }

    /** Return a stable graph identifier for an action node. */
    actionIdentifier(actionNode) {
      const actionName = String(actionNode.label || 'Action').replace(/Action$/, '');
      return 'live-action:' + actionName.toLowerCase();
    }

    /** Convert an action class name into a readable label. */
    actionLabel(actionName) {
      return String(actionName || 'Action')
        .replace(/Action$/, '')
        .replace(/([a-z])([A-Z])/g, '$1 $2');
    }

    /** Convert an arm enum value into its lowercase side name. */
    armName(arm) {
      const parts = String(arm).split('.');
      return parts[parts.length - 1].toLowerCase();
    }

    /** Convert a mesh basename into a readable object label. */
    objectLabel(objectName) {
      return String(objectName).replace(/\.[^.]+$/, '').replace(/_/g, ' ');
    }

    /** Return the signature used to detect structural live-graph changes. */
    structureSignature() {
      const objects = (this.bridgeInformation.objects || []).slice().sort().join('|');
      return [
        this.planSnapshot.sig || '',
        this.bridgeInformation.robot || '',
        objects,
      ].join('::');
    }
  }

  root.LiveKnowledgeGraph = LiveKnowledgeGraph;
})(window);
