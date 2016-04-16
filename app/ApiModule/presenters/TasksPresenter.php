<?php

namespace ApiModule;

/**
 * TasksPresenter is used for serving list of task in experiment from REST API
 */
class TasksPresenter extends BasePresenter {

	private $tasksModel;
	private $experimentsModel;

	public function __construct( \Nette\Http\Request $httpRequest, \Tasks $tasksModel, \Experiments $experimentsModel ) {
		parent::__construct( $httpRequest );
		$this->tasksModel = $tasksModel;
		$this->experimentsModel = $experimentsModel;
	}

	public function renderDefault( $experimentId ) {
		$parameters = $this->context->getParameters();

		$response = array();
		$response[ 'tasks' ] = array();
		foreach( $this->tasksModel->getTasks( $experimentId ) as $task ) {
			$taskResponse[ 'id' ] = $task->id;
			$taskResponse[ 'name' ] = $task->name;
			$taskResponse[ 'description' ] = $task->description;
			if( call_user_func( $this->canTaskBeRemoved, $task ) ) {
				$taskResponse[ 'show_administration' ] = TRUE;
				$taskResponse[ 'edit_link' ] = $this->link( ':Tasks:edit', $task->id );
				$taskResponse[ 'delete_link' ] = $this->link( ':Tasks:delete', $task->id );
			} else {
				$taskResponse[ 'show_administration' ] = FALSE;
			}

			$response[ 'tasks' ][ $task->id ] = $taskResponse;
		}

		$this->sendResponse( new \Nette\Application\Responses\JsonResponse( $response ) );
	}

	public function renderUpload() {
		if ( !call_user_func( $this->isInsertAllowed ) ) {
			throw new \Nette\Security\AuthenticationException();
		}

		$name = $this->getPostParameter( 'name' );
		$url_key = \Nette\Utils\Strings::webalize( $name );
		$description = $this->getPostParameter( 'description' );
		$experiment_id = $this->getPostParameter( 'experiment_id' );
		$translation = $this->getPostFile( 'translation' );

		$data = array(
			'name' => $name,
			'description' => $description,
			'url_key' => $url_key,
			'experiments_id' => $experiment_id,
			'created_by' => $this->user->getId(),
		);

		$experiment = $this->experimentsModel->getExperimentById( $experiment_id );
		$path = __DIR__ . '/../../../data/' . $experiment->url_key . '/' . $url_key . '/';
		$translation->move( $path . 'translation.txt' );
		file_put_contents( $path . 'config.neon', "name: $name\ndescription: $description\nurl_key: $url_key" );

		$response = array( 'task_id' => $this->tasksModel->saveTask( $data ) );

		if ( $this->getPostParameter( 'redirect', False ) ) {
			$this->flashMessage( "Task was successfully uploaded. It will appear in this list once it is imported.", "success" );
			$this->redirect( ":Tasks:list", $experiment_id );
		} else {
			$this->sendResponse( new \Nette\Application\Responses\JsonResponse( $response ) );
		}
	}

}
